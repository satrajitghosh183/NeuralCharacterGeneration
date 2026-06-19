#include <ncg/io/npy.hpp>

#include <ncg/core/error.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace ncg::io {
namespace {

at::ScalarType dtype_from_descr(const std::string& descr) {
  // Endianness char is '<', '|', or '=' (little/native — we only support little/native).
  NCG_CHECK(descr.size() >= 3, "npy: bad descr '{}'", descr);
  NCG_CHECK(descr[0] != '>', "npy: big-endian data not supported ('{}')", descr);
  const std::string kind = descr.substr(1);
  if (kind == "f4") return at::kFloat;
  if (kind == "f8") return at::kDouble;
  if (kind == "f2") return at::kHalf;
  if (kind == "i8") return at::kLong;
  if (kind == "i4") return at::kInt;
  if (kind == "i2") return at::kShort;
  if (kind == "i1") return at::kChar;
  if (kind == "u1") return at::kByte;
  if (kind == "b1") return at::kBool;
  NCG_THROW("npy: unsupported descr '{}'", descr);
}

std::string descr_from_dtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "<f4";
    case at::kDouble: return "<f8";
    case at::kHalf: return "<f2";
    case at::kLong: return "<i8";
    case at::kInt: return "<i4";
    case at::kShort: return "<i2";
    case at::kChar: return "|i1";
    case at::kByte: return "|u1";
    case at::kBool: return "|b1";
    default: NCG_THROW("npy: cannot serialize dtype {}", c10::toString(t));
  }
}

std::string extract_field(const std::string& header, const std::string& key) {
  const size_t k = header.find("'" + key + "'");
  NCG_CHECK(k != std::string::npos, "npy: header missing '{}'", key);
  const size_t colon = header.find(':', k);
  return header.substr(colon + 1);
}

}  // namespace

Tensor load_npy(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  NCG_CHECK(is.good(), "load_npy: cannot open '{}'", path);

  char magic[6];
  is.read(magic, 6);
  NCG_CHECK(std::memcmp(magic, "\x93NUMPY", 6) == 0, "load_npy: '{}' is not a .npy file", path);

  uint8_t major = 0;
  uint8_t minor = 0;
  is.read(reinterpret_cast<char*>(&major), 1);
  is.read(reinterpret_cast<char*>(&minor), 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t len16 = 0;
    is.read(reinterpret_cast<char*>(&len16), 2);
    header_len = len16;
  } else {
    is.read(reinterpret_cast<char*>(&header_len), 4);
  }

  std::string header(header_len, '\0');
  is.read(header.data(), header_len);

  // descr
  std::string descr_field = extract_field(header, "descr");
  const size_t q1 = descr_field.find('\'');
  const size_t q2 = descr_field.find('\'', q1 + 1);
  const std::string descr = descr_field.substr(q1 + 1, q2 - q1 - 1);
  const at::ScalarType dtype = dtype_from_descr(descr);

  // fortran_order
  const std::string fo = extract_field(header, "fortran_order");
  NCG_CHECK(fo.find("True") == std::string::npos, "load_npy: fortran_order not supported");

  // shape
  const std::string shape_field = extract_field(header, "shape");
  const size_t lp = shape_field.find('(');
  const size_t rp = shape_field.find(')', lp);
  const std::string inside = shape_field.substr(lp + 1, rp - lp - 1);
  std::vector<int64_t> shape;
  {
    std::string num;
    for (char ch : inside) {
      if (ch == ',' || ch == ' ') {
        if (!num.empty()) {
          shape.push_back(std::stoll(num));
          num.clear();
        }
      } else {
        num += ch;
      }
    }
    if (!num.empty()) shape.push_back(std::stoll(num));
  }

  int64_t numel = 1;
  for (int64_t d : shape) numel *= d;

  Tensor out = at::empty(at::IntArrayRef(shape), at::TensorOptions().dtype(dtype).device(at::kCPU));
  const int64_t nbytes = numel * out.element_size();
  is.read(static_cast<char*>(out.data_ptr()), nbytes);
  NCG_CHECK(is.good() || is.eof(), "load_npy: read error in '{}'", path);
  return out;
}

void save_npy(const std::string& path, const Tensor& t) {
  Tensor c = t.detach().to(at::kCPU).contiguous();

  std::string shape_tuple = "(";
  for (int64_t i = 0; i < c.dim(); ++i) {
    shape_tuple += std::to_string(c.size(i));
    if (c.dim() == 1 || i + 1 < c.dim()) shape_tuple += ",";
    if (i + 1 < c.dim()) shape_tuple += " ";
  }
  shape_tuple += ")";

  std::string dict = "{'descr': '" + descr_from_dtype(c.scalar_type()) +
                     "', 'fortran_order': False, 'shape': " + shape_tuple + ", }";

  // Pad so that 10 (preamble) + header + '\n' is a multiple of 64.
  size_t unpadded = 10 + dict.size() + 1;
  size_t pad = (64 - (unpadded % 64)) % 64;
  dict.append(pad, ' ');
  dict += '\n';
  const uint16_t header_len = static_cast<uint16_t>(dict.size());

  std::ofstream os(path, std::ios::binary);
  NCG_CHECK(os.good(), "save_npy: cannot open '{}'", path);
  os.write("\x93NUMPY", 6);
  const uint8_t ver[2] = {1, 0};
  os.write(reinterpret_cast<const char*>(ver), 2);
  os.write(reinterpret_cast<const char*>(&header_len), 2);
  os.write(dict.data(), static_cast<std::streamsize>(dict.size()));
  os.write(static_cast<const char*>(c.const_data_ptr()),
           static_cast<std::streamsize>(c.numel() * c.element_size()));
  NCG_CHECK(os.good(), "save_npy: write error for '{}'", path);
}

}  // namespace ncg::io
