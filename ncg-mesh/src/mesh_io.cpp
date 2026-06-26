#include <ncg/mesh/extract.hpp>

#include <ncg/core/error.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
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

void write_glb_skinned(const Tensor& vertices, const Tensor& faces, const Tensor& normals_in,
                       const Tensor& colors_in, const Tensor& joints_in, const Tensor& parents_in,
                       const Tensor& skin_weights_in, const std::string& path) {
  const auto v = vertices.to(at::kCPU, at::kFloat).contiguous();
  const auto f = faces.to(at::kCPU, at::kInt).contiguous();
  const auto jpos = joints_in.to(at::kCPU, at::kFloat).contiguous();
  const auto parents = parents_in.to(at::kCPU, at::kLong).contiguous();
  const int64_t V = v.size(0);
  const int64_t F = f.size(0);
  const int64_t J = jpos.size(0);
  const bool hasN = normals_in.defined() && normals_in.numel() > 0;
  const bool hasC = colors_in.defined() && colors_in.numel() > 0;
  const auto n = hasN ? normals_in.to(at::kCPU, at::kFloat).contiguous() : Tensor();
  const auto c = hasC ? colors_in.to(at::kCPU, at::kFloat).contiguous() : Tensor();

  // Reduce skinning to glTF's 4 influences/vertex; renormalize.
  const auto sw = skin_weights_in.to(at::kCPU, at::kFloat);
  auto topk = sw.topk(std::min<int64_t>(4, J), /*dim=*/1);
  auto wval = std::get<0>(topk).contiguous();          // [V,k]
  auto widx = std::get<1>(topk).to(at::kInt).contiguous();  // [V,k]
  if (wval.size(1) < 4) {  // pad to 4
    wval = torch::constant_pad_nd(wval, {0, 4 - wval.size(1)}, 0.0);
    widx = torch::constant_pad_nd(widx, {0, 4 - widx.size(1)}, 0);
  }
  wval = wval / wval.sum(1, true).clamp_min(1e-8F);
  const auto* wv = wval.data_ptr<float>();
  const auto* wi = widx.data_ptr<int32_t>();

  // --- BIN: pos|norm|col|JOINTS(u16x4)|WEIGHTS(f32x4)|IBM(f32x16)|indices(u32) ---
  std::vector<char> bin;
  std::vector<size_t> off;
  auto mark = [&]() { off.push_back(bin.size()); };
  mark();
  append_bytes(bin, v.data_ptr<float>(), static_cast<size_t>(V) * 3 * sizeof(float));
  if (hasN) {
    mark();
    append_bytes(bin, n.data_ptr<float>(), static_cast<size_t>(V) * 3 * sizeof(float));
  }
  if (hasC) {
    mark();
    append_bytes(bin, c.data_ptr<float>(), static_cast<size_t>(V) * 3 * sizeof(float));
  }
  const size_t jointsOff = bin.size();
  for (int64_t i = 0; i < V * 4; ++i) {
    const uint16_t u = static_cast<uint16_t>(wi[i]);
    bin.push_back(static_cast<char>(u & 0xff));
    bin.push_back(static_cast<char>((u >> 8) & 0xff));
  }
  const size_t weightsOff = bin.size();
  append_bytes(bin, wv, static_cast<size_t>(V) * 4 * sizeof(float));
  // Inverse bind matrices: rest rotation is identity, so IBM = translate(-joint_world).
  const size_t ibmOff = bin.size();
  const auto* jp = jpos.data_ptr<float>();
  for (int64_t j = 0; j < J; ++j) {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                   -jp[j * 3 + 0], -jp[j * 3 + 1], -jp[j * 3 + 2], 1};  // column-major
    append_bytes(bin, m, sizeof(m));
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

  // Joint node local translations + child lists from the parent hierarchy.
  const auto* pp = parents.data_ptr<int64_t>();
  std::vector<std::vector<int64_t>> children(static_cast<size_t>(J));
  for (int64_t j = 1; j < J; ++j) children[static_cast<size_t>(pp[j])].push_back(j);

  std::ostringstream js;
  js << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"NeuralCharGen\"},\"scene\":0,"
     << "\"scenes\":[{\"nodes\":[0," << J << "]}],\"nodes\":[";
  for (int64_t j = 0; j < J; ++j) {
    const int64_t par = (j == 0) ? -1 : pp[j];
    const float lx = jp[j * 3 + 0] - (par < 0 ? 0.0F : jp[par * 3 + 0]);
    const float ly = jp[j * 3 + 1] - (par < 0 ? 0.0F : jp[par * 3 + 1]);
    const float lz = jp[j * 3 + 2] - (par < 0 ? 0.0F : jp[par * 3 + 2]);
    js << "{\"translation\":[" << lx << "," << ly << "," << lz << "]";
    if (!children[static_cast<size_t>(j)].empty()) {
      js << ",\"children\":[";
      for (size_t k = 0; k < children[static_cast<size_t>(j)].size(); ++k)
        js << (k ? "," : "") << children[static_cast<size_t>(j)][k];
      js << "]";
    }
    js << "},";
  }
  js << "{\"mesh\":0,\"skin\":0}],";  // node index J = the skinned mesh

  // bufferViews + accessors (order: pos,[norm],[col],joints,weights,ibm,indices)
  int bv = 0;
  std::ostringstream bvs;
  std::ostringstream accs;
  auto add = [&](size_t boff, int64_t bytes, int target) {
    bvs << (bv ? "," : "") << "{\"buffer\":0,\"byteOffset\":" << boff << ",\"byteLength\":" << bytes
        << (target ? (",\"target\":" + std::to_string(target)) : "") << "}";
    return bv++;
  };
  int oi = 0;
  const int bvPos = add(off[oi++], V * 12, 34962);
  const int bvN = hasN ? add(off[oi++], V * 12, 34962) : -1;
  const int bvC = hasC ? add(off[oi++], V * 12, 34962) : -1;
  const int bvJ = add(jointsOff, V * 8, 34962);
  const int bvW = add(weightsOff, V * 16, 34962);
  const int bvIBM = add(ibmOff, J * 64, 0);
  const int bvIdx = add(idxOff, F * 12, 34963);

  accs << "{\"bufferView\":" << bvPos << ",\"componentType\":5126,\"count\":" << V
       << ",\"type\":\"VEC3\",\"min\":[" << mn[0] << "," << mn[1] << "," << mn[2] << "],\"max\":["
       << mx[0] << "," << mx[1] << "," << mx[2] << "]}";
  int acc = 1;
  int accN = -1, accC = -1;
  if (hasN) {
    accN = acc++;
    accs << ",{\"bufferView\":" << bvN << ",\"componentType\":5126,\"count\":" << V
         << ",\"type\":\"VEC3\"}";
  }
  if (hasC) {
    accC = acc++;
    accs << ",{\"bufferView\":" << bvC << ",\"componentType\":5126,\"count\":" << V
         << ",\"type\":\"VEC3\"}";
  }
  const int accJ = acc++;
  accs << ",{\"bufferView\":" << bvJ << ",\"componentType\":5123,\"count\":" << V
       << ",\"type\":\"VEC4\"}";
  const int accW = acc++;
  accs << ",{\"bufferView\":" << bvW << ",\"componentType\":5126,\"count\":" << V
       << ",\"type\":\"VEC4\"}";
  const int accIBM = acc++;
  accs << ",{\"bufferView\":" << bvIBM << ",\"componentType\":5126,\"count\":" << J
       << ",\"type\":\"MAT4\"}";
  const int accIdx = acc++;
  accs << ",{\"bufferView\":" << bvIdx << ",\"componentType\":5125,\"count\":" << F * 3
       << ",\"type\":\"SCALAR\"}";

  js << "\"bufferViews\":[" << bvs.str() << "],\"accessors\":[" << accs.str() << "],";
  js << "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0";
  if (hasN) js << ",\"NORMAL\":" << accN;
  if (hasC) js << ",\"COLOR_0\":" << accC;
  js << ",\"JOINTS_0\":" << accJ << ",\"WEIGHTS_0\":" << accW << "},\"indices\":" << accIdx
     << ",\"material\":0}]}],";
  js << "\"skins\":[{\"inverseBindMatrices\":" << accIBM << ",\"skeleton\":0,\"joints\":[";
  for (int64_t j = 0; j < J; ++j) js << (j ? "," : "") << j;
  js << "]}],\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,1],"
        "\"metallicFactor\":0,\"roughnessFactor\":1}}],\"buffers\":[{\"byteLength\":"
     << bin.size() << "}]}";
  std::string json = js.str();
  while (json.size() % 4 != 0) json.push_back(' ');

  std::vector<char> glb;
  put_u32(glb, 0x46546C67);
  put_u32(glb, 2);
  put_u32(glb, 0);
  put_u32(glb, static_cast<uint32_t>(json.size()));
  put_u32(glb, 0x4E4F534A);
  append_bytes(glb, json.data(), json.size());
  put_u32(glb, static_cast<uint32_t>(bin.size()));
  put_u32(glb, 0x004E4942);
  append_bytes(glb, bin.data(), bin.size());
  const uint32_t total = static_cast<uint32_t>(glb.size());
  glb[8] = static_cast<char>(total & 0xff);
  glb[9] = static_cast<char>((total >> 8) & 0xff);
  glb[10] = static_cast<char>((total >> 16) & 0xff);
  glb[11] = static_cast<char>((total >> 24) & 0xff);

  std::ofstream os(path, std::ios::binary);
  NCG_CHECK(os.good(), "write_glb_skinned: cannot open '{}'", path);
  os.write(glb.data(), static_cast<std::streamsize>(glb.size()));
}

void write_glb_textured(const Tensor& vertices, const Tensor& faces_in, const Tensor& normals_in,
                        const Tensor& uv_coords_in, const Tensor& uv_faces_in, const Tensor& joints_in,
                        const Tensor& parents_in, const Tensor& skin_weights_in,
                        const std::string& texture_png_path, const std::string& path,
                        const std::string& normal_png_path, const Tensor& rot_quats,
                        const Tensor& times) {
  const bool hasAnim = rot_quats.defined() && rot_quats.numel() > 0 && times.defined();
  const auto quats = hasAnim ? rot_quats.to(at::kCPU, at::kFloat).contiguous() : Tensor();  // [T,J,4]
  const auto ts = hasAnim ? times.to(at::kCPU, at::kFloat).contiguous() : Tensor();          // [T]
  const int64_t Tn = hasAnim ? ts.size(0) : 0;
  const auto vsrc = vertices.to(at::kCPU, at::kFloat).contiguous();
  const auto fgeo = faces_in.to(at::kCPU, at::kInt).contiguous();   // [F,3] geometry vertex idx
  const auto fuv = uv_faces_in.to(at::kCPU, at::kInt).contiguous(); // [F,3] uv vertex idx
  const auto uvc = uv_coords_in.to(at::kCPU, at::kFloat).contiguous();
  const bool hasN = normals_in.defined() && normals_in.numel() > 0;
  const auto nsrc = hasN ? normals_in.to(at::kCPU, at::kFloat).contiguous() : Tensor();
  const auto jpos = joints_in.to(at::kCPU, at::kFloat).contiguous();
  const auto parents = parents_in.to(at::kCPU, at::kLong).contiguous();
  const int64_t J = jpos.size(0), F = fgeo.size(0), n_uv = uvc.size(0);

  auto topk = skin_weights_in.to(at::kCPU, at::kFloat).topk(std::min<int64_t>(4, J), 1);
  auto wval = std::get<0>(topk).contiguous();
  auto widx = std::get<1>(topk).to(at::kInt).contiguous();
  if (wval.size(1) < 4) {
    wval = torch::constant_pad_nd(wval, {0, 4 - wval.size(1)}, 0.0);
    widx = torch::constant_pad_nd(widx, {0, 4 - widx.size(1)}, 0);
  }
  wval = wval / wval.sum(1, true).clamp_min(1e-8F);
  const auto* vp = vsrc.data_ptr<float>();
  const auto* np = hasN ? nsrc.data_ptr<float>() : nullptr;
  const auto* uvp = uvc.data_ptr<float>();
  const auto* fg = fgeo.data_ptr<int32_t>();
  const auto* fu = fuv.data_ptr<int32_t>();
  const auto* wv = wval.data_ptr<float>();
  const auto* wi = widx.data_ptr<int32_t>();

  // Unweld by (geometry-vertex, uv-vertex) pairs so each glTF vertex has one UV.
  std::unordered_map<int64_t, int32_t> remap;
  std::vector<float> pos, nor, tex, wgt;
  std::vector<uint16_t> jnt;
  std::vector<uint32_t> idx;
  auto getv = [&](int32_t gv, int32_t uv) -> int32_t {
    const int64_t key = static_cast<int64_t>(gv) * n_uv + uv;
    auto it = remap.find(key);
    if (it != remap.end()) return it->second;
    const int32_t ni = static_cast<int32_t>(pos.size() / 3);
    remap[key] = ni;
    pos.insert(pos.end(), {vp[gv * 3], vp[gv * 3 + 1], vp[gv * 3 + 2]});
    if (hasN) nor.insert(nor.end(), {np[gv * 3], np[gv * 3 + 1], np[gv * 3 + 2]});
    tex.insert(tex.end(), {uvp[uv * 2], uvp[uv * 2 + 1]});
    for (int k = 0; k < 4; ++k) jnt.push_back(static_cast<uint16_t>(wi[gv * 4 + k]));
    for (int k = 0; k < 4; ++k) wgt.push_back(wv[gv * 4 + k]);
    return ni;
  };
  for (int64_t fi = 0; fi < F; ++fi)
    for (int k = 0; k < 3; ++k) idx.push_back(static_cast<uint32_t>(getv(fg[fi * 3 + k], fu[fi * 3 + k])));
  const int64_t V = static_cast<int64_t>(pos.size() / 3);

  auto read_file = [](const std::string& p, std::vector<char>& out) {
    std::ifstream tf(p, std::ios::binary | std::ios::ate);
    if (!tf.good()) return false;
    const auto sz = tf.tellg();
    tf.seekg(0);
    out.resize(static_cast<size_t>(sz));
    tf.read(out.data(), sz);
    return true;
  };
  std::vector<char> png, npng;
  NCG_CHECK(read_file(texture_png_path, png), "write_glb_textured: cannot read texture '{}'",
            texture_png_path);
  const bool hasNT = !normal_png_path.empty() && read_file(normal_png_path, npng) && !npng.empty();

  std::vector<char> bin;
  std::vector<size_t> off;
  off.push_back(bin.size());
  append_bytes(bin, pos.data(), pos.size() * 4);
  size_t norOff = 0;
  if (hasN) { norOff = bin.size(); append_bytes(bin, nor.data(), nor.size() * 4); }
  const size_t texOff = bin.size();
  append_bytes(bin, tex.data(), tex.size() * 4);
  const size_t jntOff = bin.size();
  for (auto u : jnt) { bin.push_back(static_cast<char>(u & 0xff)); bin.push_back(static_cast<char>((u >> 8) & 0xff)); }
  const size_t wgtOff = bin.size();
  append_bytes(bin, wgt.data(), wgt.size() * 4);
  const size_t ibmOff = bin.size();
  const auto* jp = jpos.data_ptr<float>();
  for (int64_t j = 0; j < J; ++j) {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -jp[j * 3], -jp[j * 3 + 1], -jp[j * 3 + 2], 1};
    append_bytes(bin, m, sizeof(m));
  }
  const size_t idxOff = bin.size();
  append_bytes(bin, idx.data(), idx.size() * 4);
  while (bin.size() % 4 != 0) bin.push_back(0);
  const size_t pngOff = bin.size();
  append_bytes(bin, png.data(), png.size());
  while (bin.size() % 4 != 0) bin.push_back(0);
  size_t npngOff = 0;
  if (hasNT) {
    npngOff = bin.size();
    append_bytes(bin, npng.data(), npng.size());
    while (bin.size() % 4 != 0) bin.push_back(0);
  }
  size_t timeOff = 0;
  std::vector<size_t> quatOff;
  if (hasAnim) {
    timeOff = bin.size();
    append_bytes(bin, ts.data_ptr<float>(), Tn * 4);
    while (bin.size() % 4 != 0) bin.push_back(0);
    for (int64_t j = 0; j < J; ++j) {
      const auto qj = quats.select(1, j).contiguous();  // [T,4] (x,y,z,w)
      quatOff.push_back(bin.size());
      append_bytes(bin, qj.data_ptr<float>(), Tn * 16);
    }
    while (bin.size() % 4 != 0) bin.push_back(0);
  }

  const auto posT = torch::from_blob(pos.data(), {V, 3}).clone();
  const auto vmin = std::get<0>(posT.min(0));
  const auto vmax = std::get<0>(posT.max(0));
  const auto* mn = vmin.data_ptr<float>();
  const auto* mx = vmax.data_ptr<float>();

  const auto* pp = parents.data_ptr<int64_t>();
  std::vector<std::vector<int64_t>> children(static_cast<size_t>(J));
  for (int64_t j = 1; j < J; ++j) children[static_cast<size_t>(pp[j])].push_back(j);

  std::ostringstream js;
  js << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"NeuralCharGen\"},\"scene\":0,"
     << "\"scenes\":[{\"nodes\":[0," << J << "]}],\"nodes\":[";
  for (int64_t j = 0; j < J; ++j) {
    const int64_t par = (j == 0) ? -1 : pp[j];
    const float lx = jp[j * 3] - (par < 0 ? 0.0F : jp[par * 3]);
    const float ly = jp[j * 3 + 1] - (par < 0 ? 0.0F : jp[par * 3 + 1]);
    const float lz = jp[j * 3 + 2] - (par < 0 ? 0.0F : jp[par * 3 + 2]);
    js << "{\"translation\":[" << lx << "," << ly << "," << lz << "]";
    if (!children[static_cast<size_t>(j)].empty()) {
      js << ",\"children\":[";
      for (size_t k = 0; k < children[static_cast<size_t>(j)].size(); ++k)
        js << (k ? "," : "") << children[static_cast<size_t>(j)][k];
      js << "]";
    }
    js << "},";
  }
  js << "{\"mesh\":0,\"skin\":0}],";

  int bv = 0;
  std::ostringstream bvs;
  auto add = [&](size_t boff, int64_t bytes, int target) {
    bvs << (bv ? "," : "") << "{\"buffer\":0,\"byteOffset\":" << boff << ",\"byteLength\":" << bytes
        << (target ? (",\"target\":" + std::to_string(target)) : "") << "}";
    return bv++;
  };
  const int bvPos = add(off[0], V * 12, 34962);
  const int bvN = hasN ? add(norOff, V * 12, 34962) : -1;
  const int bvT = add(texOff, V * 8, 34962);
  const int bvJ = add(jntOff, V * 8, 34962);
  const int bvW = add(wgtOff, V * 16, 34962);
  const int bvIBM = add(ibmOff, J * 64, 0);
  const int bvIdx = add(idxOff, static_cast<int64_t>(idx.size()) * 4, 34963);
  const int bvImg = add(pngOff, static_cast<int64_t>(png.size()), 0);
  const int bvImgN = hasNT ? add(npngOff, static_cast<int64_t>(npng.size()), 0) : -1;
  int bvTime = -1;
  std::vector<int> bvQuat;
  if (hasAnim) {
    bvTime = add(timeOff, Tn * 4, 0);
    for (int64_t j = 0; j < J; ++j) bvQuat.push_back(add(quatOff[static_cast<size_t>(j)], Tn * 16, 0));
  }

  std::ostringstream accs;
  accs << "{\"bufferView\":" << bvPos << ",\"componentType\":5126,\"count\":" << V
       << ",\"type\":\"VEC3\",\"min\":[" << mn[0] << "," << mn[1] << "," << mn[2] << "],\"max\":["
       << mx[0] << "," << mx[1] << "," << mx[2] << "]}";
  int acc = 1, accN = -1;
  if (hasN) { accN = acc++; accs << ",{\"bufferView\":" << bvN << ",\"componentType\":5126,\"count\":" << V << ",\"type\":\"VEC3\"}"; }
  const int accT = acc++;
  accs << ",{\"bufferView\":" << bvT << ",\"componentType\":5126,\"count\":" << V << ",\"type\":\"VEC2\"}";
  const int accJ = acc++;
  accs << ",{\"bufferView\":" << bvJ << ",\"componentType\":5123,\"count\":" << V << ",\"type\":\"VEC4\"}";
  const int accW = acc++;
  accs << ",{\"bufferView\":" << bvW << ",\"componentType\":5126,\"count\":" << V << ",\"type\":\"VEC4\"}";
  const int accIBM = acc++;
  accs << ",{\"bufferView\":" << bvIBM << ",\"componentType\":5126,\"count\":" << J << ",\"type\":\"MAT4\"}";
  const int accIdx = acc++;
  accs << ",{\"bufferView\":" << bvIdx << ",\"componentType\":5125,\"count\":" << idx.size() << ",\"type\":\"SCALAR\"}";
  int accTime = -1;
  std::vector<int> accQuat;
  if (hasAnim) {
    accTime = acc++;
    accs << ",{\"bufferView\":" << bvTime << ",\"componentType\":5126,\"count\":" << Tn
         << ",\"type\":\"SCALAR\",\"min\":[" << ts.min().item<float>() << "],\"max\":["
         << ts.max().item<float>() << "]}";
    for (int64_t j = 0; j < J; ++j) {
      accQuat.push_back(acc++);
      accs << ",{\"bufferView\":" << bvQuat[static_cast<size_t>(j)]
           << ",\"componentType\":5126,\"count\":" << Tn << ",\"type\":\"VEC4\"}";
    }
  }

  js << "\"bufferViews\":[" << bvs.str() << "],\"accessors\":[" << accs.str() << "],";
  js << "\"images\":[{\"bufferView\":" << bvImg << ",\"mimeType\":\"image/png\"}";
  if (hasNT) js << ",{\"bufferView\":" << bvImgN << ",\"mimeType\":\"image/png\"}";
  js << "],\"samplers\":[{}],\"textures\":[{\"source\":0,\"sampler\":0}";
  if (hasNT) js << ",{\"source\":1,\"sampler\":0}";
  js << "],";
  js << "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0";
  if (hasN) js << ",\"NORMAL\":" << accN;
  js << ",\"TEXCOORD_0\":" << accT << ",\"JOINTS_0\":" << accJ << ",\"WEIGHTS_0\":" << accW
     << "},\"indices\":" << accIdx << ",\"material\":0}]}],";
  js << "\"skins\":[{\"inverseBindMatrices\":" << accIBM << ",\"skeleton\":0,\"joints\":[";
  for (int64_t j = 0; j < J; ++j) js << (j ? "," : "") << j;
  js << "]}],\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0},"
        "\"metallicFactor\":0,\"roughnessFactor\":1}";
  if (hasNT) js << ",\"normalTexture\":{\"index\":1}";
  js << "}]";
  if (hasAnim) {
    js << ",\"animations\":[{\"name\":\"clip\",\"samplers\":[";
    for (int64_t j = 0; j < J; ++j)
      js << (j ? "," : "") << "{\"input\":" << accTime << ",\"output\":" << accQuat[static_cast<size_t>(j)]
         << ",\"interpolation\":\"LINEAR\"}";
    js << "],\"channels\":[";
    for (int64_t j = 0; j < J; ++j)
      js << (j ? "," : "") << "{\"sampler\":" << j << ",\"target\":{\"node\":" << j
         << ",\"path\":\"rotation\"}}";
    js << "]}]";
  }
  js << ",\"buffers\":[{\"byteLength\":" << bin.size() << "}]}";
  std::string json = js.str();
  while (json.size() % 4 != 0) json.push_back(' ');

  std::vector<char> glb;
  put_u32(glb, 0x46546C67);
  put_u32(glb, 2);
  put_u32(glb, 0);
  put_u32(glb, static_cast<uint32_t>(json.size()));
  put_u32(glb, 0x4E4F534A);
  append_bytes(glb, json.data(), json.size());
  put_u32(glb, static_cast<uint32_t>(bin.size()));
  put_u32(glb, 0x004E4942);
  append_bytes(glb, bin.data(), bin.size());
  const uint32_t total = static_cast<uint32_t>(glb.size());
  glb[8] = static_cast<char>(total & 0xff);
  glb[9] = static_cast<char>((total >> 8) & 0xff);
  glb[10] = static_cast<char>((total >> 16) & 0xff);
  glb[11] = static_cast<char>((total >> 24) & 0xff);
  std::ofstream os(path, std::ios::binary);
  NCG_CHECK(os.good(), "write_glb_textured: cannot open '{}'", path);
  os.write(glb.data(), static_cast<std::streamsize>(glb.size()));
}

void write_glb_animated(const Tensor& vertices, const Tensor& faces, const Tensor& normals,
                        const Tensor& colors, const Tensor& joints_in, const Tensor& parents_in,
                        const Tensor& skin_weights_in, const Tensor& rot_quats, const Tensor& times,
                        const std::string& path) {
  const auto v = vertices.to(at::kCPU, at::kFloat).contiguous();
  const auto nrm = normals.to(at::kCPU, at::kFloat).contiguous();
  const auto col = colors.to(at::kCPU, at::kFloat).contiguous();
  const auto f = faces.to(at::kCPU, at::kInt).contiguous();
  const auto jpos = joints_in.to(at::kCPU, at::kFloat).contiguous();
  const auto parents = parents_in.to(at::kCPU, at::kLong).contiguous();
  const auto quats = rot_quats.to(at::kCPU, at::kFloat).contiguous();   // [T,J,4]
  const auto ts = times.to(at::kCPU, at::kFloat).contiguous();          // [T]
  const int64_t V = v.size(0);
  const int64_t F = f.size(0);
  const int64_t J = jpos.size(0);
  const int64_t T = ts.size(0);
  NCG_CHECK(quats.dim() == 3 && quats.size(0) == T && quats.size(1) == J && quats.size(2) == 4,
            "write_glb_animated: rot_quats must be [T,J,4]");

  // top-4 skin weights
  const auto sw = skin_weights_in.to(at::kCPU, at::kFloat);
  auto tk = sw.topk(std::min<int64_t>(4, J), 1);
  auto wval = std::get<0>(tk).contiguous();
  auto widx = std::get<1>(tk).to(at::kInt).contiguous();
  if (wval.size(1) < 4) {
    wval = torch::constant_pad_nd(wval, {0, 4 - wval.size(1)}, 0.0);
    widx = torch::constant_pad_nd(widx, {0, 4 - widx.size(1)}, 0);
  }
  wval = wval / wval.sum(1, true).clamp_min(1e-8F);

  std::vector<char> bin;
  std::ostringstream bvs;
  std::ostringstream accs;
  int idx = 0;
  auto emit = [&](const void* data, size_t bytes, int comp, const char* type, int64_t count,
                  int target, const std::string& extra) -> int {
    const size_t off = bin.size();
    append_bytes(bin, data, bytes);
    while (bin.size() % 4 != 0) bin.push_back(0);
    bvs << (idx ? "," : "") << "{\"buffer\":0,\"byteOffset\":" << off << ",\"byteLength\":" << bytes
        << (target ? (",\"target\":" + std::to_string(target)) : "") << "}";
    accs << (idx ? "," : "") << "{\"bufferView\":" << idx << ",\"componentType\":" << comp
         << ",\"count\":" << count << ",\"type\":\"" << type << "\"" << extra << "}";
    return idx++;
  };

  const auto vmin = std::get<0>(v.min(0));
  const auto vmax = std::get<0>(v.max(0));
  const auto* mn = vmin.data_ptr<float>();
  const auto* mx = vmax.data_ptr<float>();
  std::ostringstream pos_extra;
  pos_extra << ",\"min\":[" << mn[0] << "," << mn[1] << "," << mn[2] << "],\"max\":[" << mx[0] << ","
            << mx[1] << "," << mx[2] << "]";
  const int aPos = emit(v.data_ptr<float>(), V * 12, 5126, "VEC3", V, 34962, pos_extra.str());
  const int aNrm = emit(nrm.data_ptr<float>(), V * 12, 5126, "VEC3", V, 34962, "");
  const int aCol = emit(col.data_ptr<float>(), V * 12, 5126, "VEC3", V, 34962, "");
  {
    std::vector<uint16_t> ji(static_cast<size_t>(V) * 4);
    const auto* wi = widx.data_ptr<int32_t>();
    for (size_t i = 0; i < ji.size(); ++i) ji[i] = static_cast<uint16_t>(wi[i]);
    (void)emit(ji.data(), ji.size() * 2, 5123, "VEC4", V, 34962, "");
  }
  const int aW = emit(wval.contiguous().data_ptr<float>(), V * 16, 5126, "VEC4", V, 34962, "");
  std::vector<float> ibm;
  ibm.reserve(static_cast<size_t>(J) * 16);
  const auto* jp = jpos.data_ptr<float>();
  for (int64_t j = 0; j < J; ++j) {
    const float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                         -jp[j * 3 + 0], -jp[j * 3 + 1], -jp[j * 3 + 2], 1};
    for (float x : m) ibm.push_back(x);
  }
  const int aIBM = emit(ibm.data(), ibm.size() * 4, 5126, "MAT4", J, 0, "");
  std::ostringstream t_extra;
  t_extra << ",\"min\":[" << ts.min().item<float>() << "],\"max\":[" << ts.max().item<float>() << "]";
  const int aTime = emit(ts.data_ptr<float>(), T * 4, 5126, "SCALAR", T, 0, t_extra.str());
  std::vector<int> aQuat(static_cast<size_t>(J));
  for (int64_t j = 0; j < J; ++j) {
    const auto qj = quats.select(1, j).contiguous();  // [T,4]
    aQuat[static_cast<size_t>(j)] = emit(qj.data_ptr<float>(), T * 16, 5126, "VEC4", T, 0, "");
  }
  int aIdx;
  {
    const auto* fp = f.data_ptr<int32_t>();
    std::vector<uint32_t> ind(static_cast<size_t>(F) * 3);
    for (size_t i = 0; i < ind.size(); ++i) ind[i] = static_cast<uint32_t>(fp[i]);
    aIdx = emit(ind.data(), ind.size() * 4, 5125, "SCALAR", F * 3, 34963, "");
  }

  // nodes (joints + mesh) and animation channels
  const auto* pp = parents.data_ptr<int64_t>();
  std::vector<std::vector<int64_t>> children(static_cast<size_t>(J));
  for (int64_t j = 1; j < J; ++j) children[static_cast<size_t>(pp[j])].push_back(j);
  std::ostringstream js;
  js << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"NeuralCharGen\"},\"scene\":0,"
     << "\"scenes\":[{\"nodes\":[0," << J << "]}],\"nodes\":[";
  for (int64_t j = 0; j < J; ++j) {
    const int64_t par = (j == 0) ? -1 : pp[j];
    js << "{\"translation\":[" << jp[j * 3 + 0] - (par < 0 ? 0.0F : jp[par * 3 + 0]) << ","
       << jp[j * 3 + 1] - (par < 0 ? 0.0F : jp[par * 3 + 1]) << ","
       << jp[j * 3 + 2] - (par < 0 ? 0.0F : jp[par * 3 + 2]) << "]";
    if (!children[static_cast<size_t>(j)].empty()) {
      js << ",\"children\":[";
      for (size_t k = 0; k < children[static_cast<size_t>(j)].size(); ++k)
        js << (k ? "," : "") << children[static_cast<size_t>(j)][k];
      js << "]";
    }
    js << "},";
  }
  js << "{\"mesh\":0,\"skin\":0}],";
  js << "\"bufferViews\":[" << bvs.str() << "],\"accessors\":[" << accs.str() << "],";
  js << "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":" << aPos << ",\"NORMAL\":"
     << aNrm << ",\"COLOR_0\":" << aCol << ",\"JOINTS_0\":3,\"WEIGHTS_0\":" << aW
     << "},\"indices\":" << aIdx << ",\"material\":0}]}],";
  js << "\"skins\":[{\"inverseBindMatrices\":" << aIBM << ",\"skeleton\":0,\"joints\":[";
  for (int64_t j = 0; j < J; ++j) js << (j ? "," : "") << j;
  js << "]}],\"animations\":[{\"name\":\"clip\",\"samplers\":[";
  for (int64_t j = 0; j < J; ++j)
    js << (j ? "," : "") << "{\"input\":" << aTime << ",\"output\":" << aQuat[static_cast<size_t>(j)]
       << ",\"interpolation\":\"LINEAR\"}";
  js << "],\"channels\":[";
  for (int64_t j = 0; j < J; ++j)
    js << (j ? "," : "") << "{\"sampler\":" << j << ",\"target\":{\"node\":" << j
       << ",\"path\":\"rotation\"}}";
  js << "]}],\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,1],"
        "\"metallicFactor\":0,\"roughnessFactor\":1}}],\"buffers\":[{\"byteLength\":"
     << bin.size() << "}]}";
  std::string json = js.str();
  while (json.size() % 4 != 0) json.push_back(' ');

  std::vector<char> glb;
  put_u32(glb, 0x46546C67);
  put_u32(glb, 2);
  put_u32(glb, 0);
  put_u32(glb, static_cast<uint32_t>(json.size()));
  put_u32(glb, 0x4E4F534A);
  append_bytes(glb, json.data(), json.size());
  put_u32(glb, static_cast<uint32_t>(bin.size()));
  put_u32(glb, 0x004E4942);
  append_bytes(glb, bin.data(), bin.size());
  const uint32_t total = static_cast<uint32_t>(glb.size());
  glb[8] = static_cast<char>(total & 0xff);
  glb[9] = static_cast<char>((total >> 8) & 0xff);
  glb[10] = static_cast<char>((total >> 16) & 0xff);
  glb[11] = static_cast<char>((total >> 24) & 0xff);
  std::ofstream os(path, std::ios::binary);
  NCG_CHECK(os.good(), "write_glb_animated: cannot open '{}'", path);
  os.write(glb.data(), static_cast<std::streamsize>(glb.size()));
}

void write_gaussian_ply(const recon::GaussianCloud& cloud, const std::string& path,
                        const Tensor& skin_joints, const Tensor& skin_weights) {
  cloud.validate();
  const int64_t n = cloud.size();
  NCG_CHECK(n > 0, "write_gaussian_ply: empty cloud");

  // Move to CPU float and apply the 3DGS storage conventions.
  const auto pos = cloud.positions.detach().to(at::kCPU, at::kFloat).contiguous();
  const auto col = cloud.colors.detach().to(at::kCPU, at::kFloat).clamp(0.0, 1.0);
  // f_dc: color = SH_C0 * f_dc + 0.5  ->  f_dc = (color - 0.5) / SH_C0
  constexpr float kShC0 = 0.28209479177387814F;
  const auto fdc = ((col - 0.5F) / kShC0).contiguous();
  // opacity stored pre-sigmoid; scales stored as log; quaternion normalized (w,x,y,z).
  const auto op = cloud.opacities.detach().to(at::kCPU, at::kFloat).clamp(1e-6, 1.0 - 1e-6);
  const auto op_logit = (op / (1.0 - op)).log().contiguous();
  const auto logscale =
      cloud.scales.detach().to(at::kCPU, at::kFloat).clamp_min(1e-9).log().contiguous();
  auto rot = cloud.rotations.detach().to(at::kCPU, at::kFloat);
  rot = (rot / rot.norm(2, 1, true).clamp_min(1e-8)).contiguous();

  const auto pa = pos.accessor<float, 2>();
  const auto fa = fdc.accessor<float, 2>();
  const auto oa = op_logit.accessor<float, 2>();
  const auto sa = logscale.accessor<float, 2>();
  const auto ra = rot.accessor<float, 2>();

  std::ofstream os(path, std::ios::binary);
  NCG_CHECK(os.good(), "write_gaussian_ply: cannot open '{}'", path);
  os << "ply\nformat binary_little_endian 1.0\n";
  os << "element vertex " << n << "\n";
  for (const char* prop : {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                           "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}) {
    os << "property float " << prop << "\n";
  }
  os << "end_header\n";
  auto put = [&](float v) { os.write(reinterpret_cast<const char*>(&v), sizeof(float)); };
  for (int64_t i = 0; i < n; ++i) {
    put(pa[i][0]); put(pa[i][1]); put(pa[i][2]);
    put(0.0F); put(0.0F); put(0.0F);                            // normals (unused by GS)
    put(fa[i][0]); put(fa[i][1]); put(fa[i][2]);                // SH DC color
    put(oa[i][0]);                                              // opacity (pre-sigmoid)
    put(sa[i][0]); put(sa[i][1]); put(sa[i][2]);                // log scales
    put(ra[i][0]); put(ra[i][1]); put(ra[i][2]); put(ra[i][3]); // quaternion (w,x,y,z)
  }
  os.close();

  // Optional skinning sidecar: int32 N, then per-splat 4×int32 joints + 4×float32 weights.
  if (skin_joints.defined() && skin_weights.defined() && skin_joints.numel() > 0) {
    NCG_CHECK(skin_joints.size(0) == n && skin_joints.size(1) == 4, "skin_joints must be [N,4]");
    NCG_CHECK(skin_weights.size(0) == n && skin_weights.size(1) == 4, "skin_weights must be [N,4]");
    const auto jj = skin_joints.detach().to(at::kCPU, at::kInt).contiguous();
    const auto ww = skin_weights.detach().to(at::kCPU, at::kFloat).contiguous();
    const auto ja = jj.accessor<int32_t, 2>();
    const auto wa = ww.accessor<float, 2>();
    std::ofstream ss(path + ".skin", std::ios::binary);
    NCG_CHECK(ss.good(), "write_gaussian_ply: cannot open skin sidecar");
    const int32_t ni = static_cast<int32_t>(n);
    ss.write(reinterpret_cast<const char*>(&ni), sizeof(int32_t));
    for (int64_t i = 0; i < n; ++i) {
      for (int k = 0; k < 4; ++k) {
        const int32_t j = ja[i][k];
        ss.write(reinterpret_cast<const char*>(&j), sizeof(int32_t));
      }
      for (int k = 0; k < 4; ++k) {
        const float w = wa[i][k];
        ss.write(reinterpret_cast<const char*>(&w), sizeof(float));
      }
    }
  }
}

}  // namespace ncg::mesh
