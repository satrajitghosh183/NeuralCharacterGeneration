#include <ncg/mesh/extract.hpp>

#include <ncg/core/error.hpp>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <vector>

namespace ncg::mesh {

namespace {
void put_u32(std::vector<char>& b, uint32_t v) {
  b.push_back(static_cast<char>(v & 0xff));
  b.push_back(static_cast<char>((v >> 8) & 0xff));
  b.push_back(static_cast<char>((v >> 16) & 0xff));
  b.push_back(static_cast<char>((v >> 24) & 0xff));
}
void append_bytes(std::vector<char>& b, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  b.insert(b.end(), c, c + n);
}
}  // namespace

void write_glb(const Tensor& vertices, const Tensor& faces, const Tensor& normals_in,
               const Tensor& colors_in, const std::string& path) {
  const auto v = vertices.to(at::kCPU, at::kFloat).contiguous();
  const auto f = faces.to(at::kCPU, at::kInt).contiguous();
  const int64_t V = v.size(0);
  const int64_t F = f.size(0);
  NCG_CHECK(v.dim() == 2 && v.size(1) == 3, "write_glb: vertices must be [V,3]");
  NCG_CHECK(f.dim() == 2 && f.size(1) == 3, "write_glb: faces must be [F,3]");
  const bool hasN = normals_in.defined() && normals_in.numel() > 0;
  const bool hasC = colors_in.defined() && colors_in.numel() > 0;
  const auto n = hasN ? normals_in.to(at::kCPU, at::kFloat).contiguous() : Tensor();
  const auto c = hasC ? colors_in.to(at::kCPU, at::kFloat).contiguous() : Tensor();

  // --- BIN buffer: positions | [normals] | [colors] | indices(u32) ---
  std::vector<char> bin;
  const size_t posOff = 0;
  append_bytes(bin, v.data_ptr<float>(), static_cast<size_t>(V) * 3 * sizeof(float));
  size_t normOff = 0;
  size_t colOff = 0;
  if (hasN) {
    normOff = bin.size();
    append_bytes(bin, n.data_ptr<float>(), static_cast<size_t>(V) * 3 * sizeof(float));
  }
  if (hasC) {
    colOff = bin.size();
    append_bytes(bin, c.data_ptr<float>(), static_cast<size_t>(V) * 3 * sizeof(float));
  }
  const size_t idxOff = bin.size();
  {
    const auto* fp = f.data_ptr<int32_t>();
    std::vector<uint32_t> idx(static_cast<size_t>(F) * 3);
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<uint32_t>(fp[i]);
    append_bytes(bin, idx.data(), idx.size() * sizeof(uint32_t));
  }
  while (bin.size() % 4 != 0) bin.push_back(0);

  const auto vmin = std::get<0>(v.min(0));
  const auto vmax = std::get<0>(v.max(0));
  const auto* mn = vmin.data_ptr<float>();
  const auto* mx = vmax.data_ptr<float>();

  // --- JSON ---
  const int bvIdx = 1 + (hasN ? 1 : 0) + (hasC ? 1 : 0);
  std::ostringstream js;
  js << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"NeuralCharGen\"},"
     << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],\"bufferViews\":[";
  js << "{\"buffer\":0,\"byteOffset\":" << posOff << ",\"byteLength\":" << V * 3 * 4
     << ",\"target\":34962}";
  if (hasN)
    js << ",{\"buffer\":0,\"byteOffset\":" << normOff << ",\"byteLength\":" << V * 3 * 4
       << ",\"target\":34962}";
  if (hasC)
    js << ",{\"buffer\":0,\"byteOffset\":" << colOff << ",\"byteLength\":" << V * 3 * 4
       << ",\"target\":34962}";
  js << ",{\"buffer\":0,\"byteOffset\":" << idxOff << ",\"byteLength\":" << F * 3 * 4
     << ",\"target\":34963}],\"accessors\":[";
  js << "{\"bufferView\":0,\"componentType\":5126,\"count\":" << V << ",\"type\":\"VEC3\",\"min\":["
     << mn[0] << "," << mn[1] << "," << mn[2] << "],\"max\":[" << mx[0] << "," << mx[1] << ","
     << mx[2] << "]}";
  int acc = 1;
  int accN = -1;
  int accC = -1;
  if (hasN) {
    accN = acc++;
    js << ",{\"bufferView\":1,\"componentType\":5126,\"count\":" << V << ",\"type\":\"VEC3\"}";
  }
  if (hasC) {
    accC = acc++;
    js << ",{\"bufferView\":" << (hasN ? 2 : 1) << ",\"componentType\":5126,\"count\":" << V
       << ",\"type\":\"VEC3\"}";
  }
  const int accIdx = acc++;
  js << ",{\"bufferView\":" << bvIdx << ",\"componentType\":5125,\"count\":" << F * 3
     << ",\"type\":\"SCALAR\"}],\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0";
  if (hasN) js << ",\"NORMAL\":" << accN;
  if (hasC) js << ",\"COLOR_0\":" << accC;
  js << "},\"indices\":" << accIdx
     << ",\"material\":0}]}],\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,"
        "1],\"metallicFactor\":0,\"roughnessFactor\":1}}],\"buffers\":[{\"byteLength\":"
     << bin.size() << "}]}";
  std::string json = js.str();
  while (json.size() % 4 != 0) json.push_back(' ');

  // --- assemble GLB container ---
  std::vector<char> glb;
  put_u32(glb, 0x46546C67);  // "glTF"
  put_u32(glb, 2);           // version
  put_u32(glb, 0);           // total length (patched below)
  put_u32(glb, static_cast<uint32_t>(json.size()));
  put_u32(glb, 0x4E4F534A);  // "JSON"
  append_bytes(glb, json.data(), json.size());
  put_u32(glb, static_cast<uint32_t>(bin.size()));
  put_u32(glb, 0x004E4942);  // "BIN\0"
  append_bytes(glb, bin.data(), bin.size());
  const uint32_t total = static_cast<uint32_t>(glb.size());
  glb[8] = static_cast<char>(total & 0xff);
  glb[9] = static_cast<char>((total >> 8) & 0xff);
  glb[10] = static_cast<char>((total >> 16) & 0xff);
  glb[11] = static_cast<char>((total >> 24) & 0xff);

  std::ofstream os(path, std::ios::binary);
  NCG_CHECK(os.good(), "write_glb: cannot open '{}'", path);
  os.write(glb.data(), static_cast<std::streamsize>(glb.size()));
}

void write_obj(const TriMesh& mesh, const std::string& path) {
  const auto v = mesh.vertices.to(at::kCPU, at::kFloat).contiguous();
  const auto f = mesh.faces.to(at::kCPU, at::kLong).contiguous();
  std::ofstream os(path);
  NCG_CHECK(os.good(), "write_obj: cannot open '{}'", path);

  const auto* vp = v.data_ptr<float>();
  for (int64_t i = 0; i < v.size(0); ++i) {
    os << "v " << vp[i * 3 + 0] << ' ' << vp[i * 3 + 1] << ' ' << vp[i * 3 + 2] << '\n';
  }
  const auto* fp = f.data_ptr<int64_t>();
  for (int64_t i = 0; i < f.size(0); ++i) {
    // OBJ is 1-indexed.
    os << "f " << fp[i * 3 + 0] + 1 << ' ' << fp[i * 3 + 1] + 1 << ' ' << fp[i * 3 + 2] + 1 << '\n';
  }
  NCG_CHECK(os.good(), "write_obj: write error for '{}'", path);
}

void write_ply(const TriMesh& mesh, const std::string& path) {
  const auto v = mesh.vertices.to(at::kCPU, at::kFloat).contiguous();
  const auto f = mesh.faces.to(at::kCPU, at::kLong).contiguous();
  std::ofstream os(path);
  NCG_CHECK(os.good(), "write_ply: cannot open '{}'", path);

  os << "ply\nformat ascii 1.0\n";
  os << "element vertex " << v.size(0) << "\n";
  os << "property float x\nproperty float y\nproperty float z\n";
  os << "element face " << f.size(0) << "\n";
  os << "property list uchar int vertex_indices\n";
  os << "end_header\n";

  const auto* vp = v.data_ptr<float>();
  for (int64_t i = 0; i < v.size(0); ++i) {
    os << vp[i * 3 + 0] << ' ' << vp[i * 3 + 1] << ' ' << vp[i * 3 + 2] << '\n';
  }
  const auto* fp = f.data_ptr<int64_t>();
  for (int64_t i = 0; i < f.size(0); ++i) {
    os << "3 " << fp[i * 3 + 0] << ' ' << fp[i * 3 + 1] << ' ' << fp[i * 3 + 2] << '\n';
  }
  NCG_CHECK(os.good(), "write_ply: write error for '{}'", path);
}

}  // namespace ncg::mesh
