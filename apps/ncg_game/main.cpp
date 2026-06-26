// ncg_game — "Rock Walker": a small Vulkan game that loads the rigged .glb character produced by
// `ncg_cli avatar`, stands it on a grid floor, plays its skeletal animation, and lets you walk it
// around with WASD (+ jump with simple gravity) while orbiting the camera with the mouse.
//
// Built on a machine with a display + Vulkan SDK (M4 Pro via MoltenVK, or Linux+GPU). See the
// CMakeLists for prerequisites and build steps. Render path: traditional render pass (MoltenVK
// friendly), LBS skinning in the vertex shader, joint matrices evaluated CPU-side per frame.
#include <vk_mem_alloc.h>
#include <VkBootstrap.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <tiny_gltf.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(x)                                                      \
  do {                                                                   \
    VkResult err__ = (x);                                                \
    if (err__ != VK_SUCCESS) {                                           \
      std::fprintf(stderr, "Vulkan error %d at %s:%d\n", err__, __FILE__, __LINE__); \
      std::abort();                                                      \
    }                                                                    \
  } while (0)

namespace {

constexpr int kMaxFrames = 2;

struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec3 color;
  glm::uvec4 joints;
  glm::vec4 weights;
};

struct CamUBO {
  glm::mat4 view;
  glm::mat4 proj;
  glm::vec4 lightDir;
  glm::vec4 camPos;
};

// ---- glTF skeleton + animation (rotation channels, as written by ncg-mesh) ----
struct Node {
  glm::vec3 t{0};
  glm::quat r{1, 0, 0, 0};
  glm::vec3 s{1};
  std::vector<int> children;
  glm::mat4 local() const {
    return glm::translate(glm::mat4(1), t) * glm::mat4_cast(r) * glm::scale(glm::mat4(1), s);
  }
};
struct RotTrack {
  int node;
  std::vector<float> times;
  std::vector<glm::quat> rots;
};

struct Model {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  std::vector<Node> nodes;
  std::vector<int> roots;
  std::vector<int> jointNodes;          // skin.joints
  std::vector<glm::mat4> invBind;       // inverseBindMatrices
  int meshNode = -1;
  std::vector<RotTrack> tracks;
  float duration = 0.0f;
  float footY = 0.0f;  // lowest vertex in bind pose -> ground offset
};

template <typename T>
const T* accessorData(const tinygltf::Model& m, int accessorIdx, size_t& count, int& comps) {
  const auto& acc = m.accessors[accessorIdx];
  const auto& bv = m.bufferViews[acc.bufferView];
  const auto& buf = m.buffers[bv.buffer];
  count = acc.count;
  comps = tinygltf::GetNumComponentsInType(acc.type);
  const size_t off = acc.byteOffset + bv.byteOffset;
  return reinterpret_cast<const T*>(buf.data.data() + off);
}

Model loadGLB(const std::string& path) {
  tinygltf::TinyGLTF ctx;
  tinygltf::Model gm;
  std::string err, warn;
  bool ok = ctx.LoadBinaryFromFile(&gm, &err, &warn, path);
  if (!warn.empty()) std::fprintf(stderr, "glTF warn: %s\n", warn.c_str());
  if (!ok) throw std::runtime_error("failed to load glb: " + err);

  Model out;
  out.nodes.resize(gm.nodes.size());
  for (size_t i = 0; i < gm.nodes.size(); ++i) {
    const auto& n = gm.nodes[i];
    Node nd;
    if (n.translation.size() == 3) nd.t = {(float)n.translation[0], (float)n.translation[1], (float)n.translation[2]};
    if (n.scale.size() == 3) nd.s = {(float)n.scale[0], (float)n.scale[1], (float)n.scale[2]};
    if (n.rotation.size() == 4)
      nd.r = glm::quat((float)n.rotation[3], (float)n.rotation[0], (float)n.rotation[1], (float)n.rotation[2]);
    for (int c : n.children) nd.children.push_back(c);
    out.nodes[i] = nd;
    if (n.mesh >= 0) out.meshNode = (int)i;
  }
  const auto& scene = gm.scenes[gm.defaultScene >= 0 ? gm.defaultScene : 0];
  for (int r : scene.nodes) out.roots.push_back(r);

  // Mesh (first primitive of the mesh-bearing node).
  const auto& mesh = gm.meshes[gm.nodes[out.meshNode].mesh];
  const auto& prim = mesh.primitives[0];
  size_t cnt; int comps;
  const float* pos = accessorData<float>(gm, prim.attributes.at("POSITION"), cnt, comps);
  const size_t nv = cnt;
  out.vertices.resize(nv);
  for (size_t i = 0; i < nv; ++i) out.vertices[i].pos = {pos[3*i], pos[3*i+1], pos[3*i+2]};
  if (prim.attributes.count("NORMAL")) {
    const float* nn = accessorData<float>(gm, prim.attributes.at("NORMAL"), cnt, comps);
    for (size_t i = 0; i < nv; ++i) out.vertices[i].normal = {nn[3*i], nn[3*i+1], nn[3*i+2]};
  }
  if (prim.attributes.count("COLOR_0")) {
    const auto& acc = gm.accessors[prim.attributes.at("COLOR_0")];
    const float* cc = accessorData<float>(gm, prim.attributes.at("COLOR_0"), cnt, comps);
    for (size_t i = 0; i < nv; ++i)
      out.vertices[i].color = {cc[comps*i], cc[comps*i+1], cc[comps*i+2]};
    (void)acc;
  } else {
    for (auto& v : out.vertices) v.color = {0.8f, 0.75f, 0.7f};
  }
  if (prim.attributes.count("JOINTS_0")) {
    const auto& acc = gm.accessors[prim.attributes.at("JOINTS_0")];
    size_t jc; int jcomp;
    if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
      const uint8_t* jj = accessorData<uint8_t>(gm, prim.attributes.at("JOINTS_0"), jc, jcomp);
      for (size_t i = 0; i < nv; ++i) out.vertices[i].joints = {jj[4*i], jj[4*i+1], jj[4*i+2], jj[4*i+3]};
    } else {
      const uint16_t* jj = accessorData<uint16_t>(gm, prim.attributes.at("JOINTS_0"), jc, jcomp);
      for (size_t i = 0; i < nv; ++i) out.vertices[i].joints = {jj[4*i], jj[4*i+1], jj[4*i+2], jj[4*i+3]};
    }
  }
  if (prim.attributes.count("WEIGHTS_0")) {
    const float* ww = accessorData<float>(gm, prim.attributes.at("WEIGHTS_0"), cnt, comps);
    for (size_t i = 0; i < nv; ++i) out.vertices[i].weights = {ww[4*i], ww[4*i+1], ww[4*i+2], ww[4*i+3]};
  } else {
    for (auto& v : out.vertices) v.weights = {1, 0, 0, 0};
  }
  // Indices.
  {
    const auto& acc = gm.accessors[prim.indices];
    size_t ic; int icomp;
    if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
      const uint32_t* id = accessorData<uint32_t>(gm, prim.indices, ic, icomp);
      out.indices.assign(id, id + ic);
    } else {
      const uint16_t* id = accessorData<uint16_t>(gm, prim.indices, ic, icomp);
      out.indices.resize(ic);
      for (size_t i = 0; i < ic; ++i) out.indices[i] = id[i];
    }
  }
  // Skin.
  if (!gm.skins.empty()) {
    const auto& sk = gm.skins[0];
    for (int j : sk.joints) out.jointNodes.push_back(j);
    size_t mc; int mcomp;
    const float* ib = accessorData<float>(gm, sk.inverseBindMatrices, mc, mcomp);
    out.invBind.resize(mc);
    for (size_t i = 0; i < mc; ++i) out.invBind[i] = glm::make_mat4(ib + 16 * i);
  }
  // Animation (rotation channels).
  if (!gm.animations.empty()) {
    const auto& an = gm.animations[0];
    for (const auto& ch : an.channels) {
      if (ch.target_path != "rotation") continue;
      const auto& smp = an.samplers[ch.sampler];
      RotTrack tr; tr.node = ch.target_node;
      size_t tc; int tcomp;
      const float* tt = accessorData<float>(gm, smp.input, tc, tcomp);
      tr.times.assign(tt, tt + tc);
      const float* rr = accessorData<float>(gm, smp.output, tc, tcomp);
      for (size_t i = 0; i < tc; ++i)
        tr.rots.push_back(glm::quat(rr[4*i+3], rr[4*i+0], rr[4*i+1], rr[4*i+2]));
      if (!tr.times.empty()) out.duration = std::max(out.duration, tr.times.back());
      out.tracks.push_back(std::move(tr));
    }
  }
  return out;
}

void evalGlobals(const Model& m, std::vector<glm::quat>& animRot, std::vector<glm::mat4>& global) {
  global.assign(m.nodes.size(), glm::mat4(1));
  std::vector<Node> nodes = m.nodes;
  for (size_t i = 0; i < nodes.size(); ++i)
    if (animRot.size() == nodes.size()) nodes[i].r = animRot[i];
  // Recursive traversal from roots.
  std::function<void(int, const glm::mat4&)> walk = [&](int idx, const glm::mat4& parent) {
    glm::mat4 g = parent * nodes[idx].local();
    global[idx] = g;
    for (int c : nodes[idx].children) walk(c, g);
  };
  for (int r : m.roots) walk(r, glm::mat4(1));
}

// Joint matrices for the shader: inv(meshNodeGlobal) * jointGlobal * inverseBind.
std::vector<glm::mat4> jointMatrices(const Model& m, float t) {
  std::vector<glm::quat> animRot(m.nodes.size());
  for (size_t i = 0; i < m.nodes.size(); ++i) animRot[i] = m.nodes[i].r;
  for (const auto& tr : m.tracks) {
    if (tr.times.empty()) continue;
    float tt = m.duration > 0 ? std::fmod(t, m.duration) : 0.0f;
    size_t k = 0;
    while (k + 1 < tr.times.size() && tr.times[k + 1] < tt) ++k;
    if (k + 1 < tr.times.size()) {
      float a = (tt - tr.times[k]) / std::max(1e-6f, tr.times[k + 1] - tr.times[k]);
      animRot[tr.node] = glm::slerp(tr.rots[k], tr.rots[k + 1], glm::clamp(a, 0.0f, 1.0f));
    } else {
      animRot[tr.node] = tr.rots.back();
    }
  }
  std::vector<glm::mat4> global;
  evalGlobals(m, animRot, global);
  glm::mat4 invMesh = m.meshNode >= 0 ? glm::inverse(global[m.meshNode]) : glm::mat4(1);
  std::vector<glm::mat4> jm(m.jointNodes.size(), glm::mat4(1));
  for (size_t i = 0; i < m.jointNodes.size(); ++i)
    jm[i] = invMesh * global[m.jointNodes[i]] * m.invBind[i];
  return jm;
}

std::vector<char> readFile(const std::string& p) {
  std::ifstream f(p, std::ios::ate | std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  size_t sz = (size_t)f.tellg();
  std::vector<char> b(sz);
  f.seekg(0); f.read(b.data(), sz);
  return b;
}
VkShaderModule loadShader(VkDevice dev, const std::string& spv) {
  auto code = readFile(spv);
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
  VkShaderModule mod;
  VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &mod));
  return mod;
}

// ---------------------------------------------------------------------------
struct App {
  GLFWwindow* window = nullptr;
  vkb::Instance vkbInst;
  vkb::Device vkbDev;
  VkSurfaceKHR surface{};
  VkDevice dev{};
  VkPhysicalDevice phys{};
  VkQueue gfxQueue{}, presentQueue{};
  uint32_t gfxQueueIdx = 0;
  VmaAllocator allocator{};
  vkb::Swapchain swap;
  std::vector<VkImage> swapImages;
  std::vector<VkImageView> swapViews;
  VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
  VkImage depthImage{}; VmaAllocation depthAlloc{}; VkImageView depthView{};
  VkRenderPass renderPass{};
  std::vector<VkFramebuffer> framebuffers;
  VkDescriptorSetLayout setLayout{};
  VkPipelineLayout pipeLayout{};
  VkPipeline meshPipe{}, floorPipe{};
  VkDescriptorPool descPool{};
  VkCommandPool cmdPool{};
  std::array<VkCommandBuffer, kMaxFrames> cmd{};
  std::array<VkSemaphore, kMaxFrames> imgAvail{}, renderDone{};
  std::array<VkFence, kMaxFrames> inFlight{};

  // GPU buffers.
  VkBuffer vbuf{}, ibuf{}, floorVbuf{};
  VmaAllocation valloc{}, ialloc{}, floorAlloc{};
  std::array<VkBuffer, kMaxFrames> camBuf{}, jointBuf{};
  std::array<VmaAllocation, kMaxFrames> camAlloc{}, jointAlloc{};
  std::array<void*, kMaxFrames> camMap{}, jointMap{};
  std::array<VkDescriptorSet, kMaxFrames> descSets{};

  Model model;
  uint32_t indexCount = 0;
  size_t jointCount = 0;

  // Camera + character state.
  float yaw = 2.2f, pitch = 0.5f, dist = 4.0f;
  glm::vec3 charPos{0, 0, 0};
  float charYaw = 0.0f;
  float velY = 0.0f;
  bool onGround = true;
  double lastX = 0, lastY = 0; bool dragging = false;
  int width = 1280, height = 800;

  void run(const std::string& glb) {
    model = loadGLB(glb);
    indexCount = (uint32_t)model.indices.size();
    jointCount = std::max<size_t>(1, model.jointNodes.size());
    // Ground the character: shift so the lowest bind-pose vertex sits on y=0.
    float minY = 1e9f;
    for (auto& v : model.vertices) minY = std::min(minY, v.pos.y);
    model.footY = minY;
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(width, height, "ncg_game — Rock Walker", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int b, int a, int) {
      auto* s = (App*)glfwGetWindowUserPointer(w);
      if (b == GLFW_MOUSE_BUTTON_LEFT) {
        s->dragging = (a == GLFW_PRESS);
        glfwGetCursorPos(w, &s->lastX, &s->lastY);
      }
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
      auto* s = (App*)glfwGetWindowUserPointer(w);
      if (s->dragging) {
        s->yaw += float(x - s->lastX) * 0.005f;
        s->pitch = glm::clamp(s->pitch + float(y - s->lastY) * 0.005f, -1.4f, 1.4f);
      }
      s->lastX = x; s->lastY = y;
    });
    glfwSetScrollCallback(window, [](GLFWwindow* w, double, double dy) {
      auto* s = (App*)glfwGetWindowUserPointer(w);
      s->dist = glm::clamp(s->dist - float(dy) * 0.4f, 1.5f, 15.0f);
    });
  }

  uint32_t findMemoryType(uint32_t, VkMemoryPropertyFlags) { return 0; }  // VMA handles this

  void createBufferVMA(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage mu,
                       VkBuffer& buf, VmaAllocation& alloc, bool mapped, void** mapPtr) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo ai{};
    ai.usage = mu;
    if (mapped) ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(allocator, &bi, &ai, &buf, &alloc, &info));
    if (mapped && mapPtr) *mapPtr = info.pMappedData;
  }

  void uploadViaStaging(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                        VkBuffer& buf, VmaAllocation& alloc) {
    VkBuffer staging; VmaAllocation sa; void* m = nullptr;
    createBufferVMA(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                    staging, sa, true, &m);
    std::memcpy(m, data, size);
    createBufferVMA(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, buf, alloc, false, nullptr);
    // one-shot copy
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = cmdPool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    VkCommandBuffer c; vkAllocateCommandBuffers(dev, &ca, &c);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(c, &bi);
    VkBufferCopy cp{0, 0, size}; vkCmdCopyBuffer(c, staging, buf, 1, &cp);
    vkEndCommandBuffer(c);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &c;
    vkQueueSubmit(gfxQueue, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(gfxQueue);
    vkFreeCommandBuffers(dev, cmdPool, 1, &c);
    vmaDestroyBuffer(allocator, staging, sa);
  }

  void initVulkan() {
    vkb::InstanceBuilder ib;
    auto inst = ib.set_app_name("ncg_game").request_validation_layers(false)
                  .require_api_version(1, 1, 0).use_default_debug_messenger().build();
    if (!inst) throw std::runtime_error("instance: " + inst.error().message());
    vkbInst = inst.value();
    VK_CHECK(glfwCreateWindowSurface(vkbInst.instance, window, nullptr, &surface));
    vkb::PhysicalDeviceSelector sel(vkbInst);
    auto pd = sel.set_surface(surface).set_minimum_version(1, 1)
                 .require_present().select();
    if (!pd) throw std::runtime_error("phys: " + pd.error().message());
    vkb::DeviceBuilder db(pd.value());
    auto dv = db.build();
    if (!dv) throw std::runtime_error("device: " + dv.error().message());
    vkbDev = dv.value();
    dev = vkbDev.device; phys = pd.value().physical_device;
    gfxQueue = vkbDev.get_queue(vkb::QueueType::graphics).value();
    presentQueue = vkbDev.get_queue(vkb::QueueType::present).value();
    gfxQueueIdx = vkbDev.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo aci{};
    aci.physicalDevice = phys; aci.device = dev; aci.instance = vkbInst.instance;
    aci.vulkanApiVersion = VK_API_VERSION_1_1;
    VK_CHECK(vmaCreateAllocator(&aci, &allocator));

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = gfxQueueIdx;
    VK_CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &cmdPool));

    createSwapchain();
    createRenderPass();
    createDepth();
    createFramebuffers();
    createDescriptors();
    createPipelines();
    uploadGeometry();
    createSync();
  }

  void createSwapchain() {
    vkb::SwapchainBuilder sb(vkbDev, surface);
    auto sc = sb.set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
                .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                .set_desired_extent(width, height).build();
    if (!sc) throw std::runtime_error("swapchain: " + sc.error().message());
    swap = sc.value();
    swapImages = swap.get_images().value();
    swapViews = swap.get_image_views().value();
    width = swap.extent.width; height = swap.extent.height;
  }

  void createRenderPass() {
    VkAttachmentDescription color{};
    color.format = swap.image_format; color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentDescription depth{};
    depth.format = depthFormat; depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1; sub.pColorAttachments = &colorRef; sub.pDepthStencilAttachment = &depthRef;
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    std::array<VkAttachmentDescription, 2> atts{color, depth};
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount = 2; rp.pAttachments = atts.data();
    rp.subpassCount = 1; rp.pSubpasses = &sub; rp.dependencyCount = 1; rp.pDependencies = &dep;
    VK_CHECK(vkCreateRenderPass(dev, &rp, nullptr, &renderPass));
  }

  void createDepth() {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = depthFormat;
    ii.extent = {(uint32_t)width, (uint32_t)height, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VmaAllocationCreateInfo ai{}; ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateImage(allocator, &ii, &ai, &depthImage, &depthAlloc, nullptr));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = depthImage; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = depthFormat;
    vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &depthView));
  }

  void createFramebuffers() {
    framebuffers.resize(swapViews.size());
    for (size_t i = 0; i < swapViews.size(); ++i) {
      std::array<VkImageView, 2> att{swapViews[i], depthView};
      VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      fi.renderPass = renderPass; fi.attachmentCount = 2; fi.pAttachments = att.data();
      fi.width = width; fi.height = height; fi.layers = 1;
      VK_CHECK(vkCreateFramebuffer(dev, &fi, nullptr, &framebuffers[i]));
    }
  }

  void createDescriptors() {
    std::array<VkDescriptorSetLayoutBinding, 2> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 2; li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &li, nullptr, &setLayout));

    VkPushConstantRange pc{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &setLayout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pipeLayout));

    std::array<VkDescriptorPoolSize, 2> ps{{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxFrames},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxFrames}}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = kMaxFrames; pi.poolSizeCount = 2; pi.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(dev, &pi, nullptr, &descPool));

    for (int i = 0; i < kMaxFrames; ++i) {
      createBufferVMA(sizeof(CamUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                      camBuf[i], camAlloc[i], true, &camMap[i]);
      createBufferVMA(sizeof(glm::mat4) * jointCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VMA_MEMORY_USAGE_AUTO_PREFER_HOST, jointBuf[i], jointAlloc[i], true, &jointMap[i]);
      VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      da.descriptorPool = descPool; da.descriptorSetCount = 1; da.pSetLayouts = &setLayout;
      VK_CHECK(vkAllocateDescriptorSets(dev, &da, &descSets[i]));
      VkDescriptorBufferInfo cb{camBuf[i], 0, sizeof(CamUBO)};
      VkDescriptorBufferInfo jb{jointBuf[i], 0, sizeof(glm::mat4) * jointCount};
      std::array<VkWriteDescriptorSet, 2> w{};
      w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[0].dstSet = descSets[i]; w[0].dstBinding = 0;
      w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &cb;
      w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[1].dstSet = descSets[i]; w[1].dstBinding = 1;
      w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &jb;
      vkUpdateDescriptorSets(dev, 2, w.data(), 0, nullptr);
    }
  }

  VkPipeline buildPipeline(const std::string& vs, const std::string& fs, bool meshAttribs) {
    VkShaderModule v = loadShader(dev, vs), f = loadShader(dev, fs);
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = v; stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = f; stages[1].pName = "main";

    VkVertexInputBindingDescription bind{0, meshAttribs ? (uint32_t)sizeof(Vertex) : (uint32_t)sizeof(glm::vec3),
                                         VK_VERTEX_INPUT_RATE_VERTEX};
    std::vector<VkVertexInputAttributeDescription> attrs;
    if (meshAttribs) {
      attrs = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
               {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
               {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
               {3, 0, VK_FORMAT_R32G32B32A32_UINT, offsetof(Vertex, joints)},
               {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, weights)}};
    } else {
      attrs = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
    }
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vin.vertexBindingDescriptionCount = 1; vin.pVertexBindingDescriptions = &bind;
    vin.vertexAttributeDescriptionCount = (uint32_t)attrs.size(); vin.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    std::array<VkDynamicState, 2> dyn{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds2{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds2.dynamicStateCount = 2; ds2.pDynamicStates = dyn.data();

    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2; gp.pStages = stages; gp.pVertexInputState = &vin; gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp; gp.pRasterizationState = &rs; gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &ds; gp.pColorBlendState = &cb; gp.pDynamicState = &ds2;
    gp.layout = pipeLayout; gp.renderPass = renderPass; gp.subpass = 0;
    VkPipeline pipe;
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe));
    vkDestroyShaderModule(dev, v, nullptr); vkDestroyShaderModule(dev, f, nullptr);
    return pipe;
  }

  void createPipelines() {
    meshPipe = buildPipeline(SPV_DIR "/mesh.vert.spv", SPV_DIR "/mesh.frag.spv", true);
    floorPipe = buildPipeline(SPV_DIR "/floor.vert.spv", SPV_DIR "/floor.frag.spv", false);
  }

  void uploadGeometry() {
    uploadViaStaging(model.vertices.data(), sizeof(Vertex) * model.vertices.size(),
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vbuf, valloc);
    uploadViaStaging(model.indices.data(), sizeof(uint32_t) * model.indices.size(),
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT, ibuf, ialloc);
    // Floor: a big quad on y=0 (two triangles).
    float e = 40.0f;
    std::vector<glm::vec3> fv{{-e, 0, -e}, {e, 0, -e}, {e, 0, e}, {-e, 0, -e}, {e, 0, e}, {-e, 0, e}};
    uploadViaStaging(fv.data(), sizeof(glm::vec3) * fv.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     floorVbuf, floorAlloc);
  }

  void createSync() {
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = cmdPool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = kMaxFrames;
    VK_CHECK(vkAllocateCommandBuffers(dev, &ca, cmd.data()));
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < kMaxFrames; ++i) {
      VK_CHECK(vkCreateSemaphore(dev, &si, nullptr, &imgAvail[i]));
      VK_CHECK(vkCreateSemaphore(dev, &si, nullptr, &renderDone[i]));
      VK_CHECK(vkCreateFence(dev, &fi, nullptr, &inFlight[i]));
    }
  }

  void updateCharacter(float dt) {
    // Movement relative to camera yaw; WASD on the xz-plane.
    glm::vec3 fwd{std::sin(yaw), 0, std::cos(yaw)};
    glm::vec3 right{fwd.z, 0, -fwd.x};
    glm::vec3 move{0};
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move -= fwd;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move += fwd;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move -= right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move += right;
    if (glm::length(move) > 1e-3f) {
      move = glm::normalize(move);
      charPos += move * 2.5f * dt;
      charYaw = std::atan2(move.x, move.z);
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && onGround) { velY = 4.0f; onGround = false; }
    // Gravity + ground collision (real, if simple).
    velY -= 9.81f * dt;
    charPos.y += velY * dt;
    if (charPos.y <= 0.0f) { charPos.y = 0.0f; velY = 0.0f; onGround = true; }
  }

  void recordAndDraw(uint32_t frame, float t) {
    vkWaitForFences(dev, 1, &inFlight[frame], VK_TRUE, UINT64_MAX);
    uint32_t imgIdx;
    VkResult acq = vkAcquireNextImageKHR(dev, swap.swapchain, UINT64_MAX, imgAvail[frame], VK_NULL_HANDLE, &imgIdx);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
    vkResetFences(dev, 1, &inFlight[frame]);

    // Update camera + joints.
    glm::vec3 target = charPos + glm::vec3(0, 1.0f, 0);
    glm::vec3 eye = target + dist * glm::vec3(std::cos(pitch) * std::sin(yaw), std::sin(pitch),
                                              std::cos(pitch) * std::cos(yaw));
    CamUBO ubo{};
    ubo.view = glm::lookAt(eye, target, glm::vec3(0, 1, 0));
    ubo.proj = glm::perspective(glm::radians(55.0f), (float)width / height, 0.05f, 200.0f);
    ubo.proj[1][1] *= -1;  // Vulkan clip-space Y flip
    ubo.lightDir = glm::vec4(glm::normalize(glm::vec3(-0.4f, -1.0f, -0.3f)), 0);
    ubo.camPos = glm::vec4(eye, 1);
    std::memcpy(camMap[frame], &ubo, sizeof(ubo));

    auto jm = jointMatrices(model, t);
    if (jm.empty()) jm.push_back(glm::mat4(1));
    std::memcpy(jointMap[frame], jm.data(), sizeof(glm::mat4) * std::min(jm.size(), jointCount));

    // Character model matrix: place on floor (foot at y=0), face heading.
    glm::mat4 m = glm::translate(glm::mat4(1), charPos + glm::vec3(0, -model.footY, 0));
    m = m * glm::rotate(glm::mat4(1), charYaw, glm::vec3(0, 1, 0));

    VkCommandBuffer c = cmd[frame];
    vkResetCommandBuffer(c, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(c, &bi);
    std::array<VkClearValue, 2> clears{};
    clears[0].color = {{0.05f, 0.06f, 0.09f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = renderPass; rp.framebuffer = framebuffers[imgIdx];
    rp.renderArea.extent = {(uint32_t)width, (uint32_t)height};
    rp.clearValueCount = 2; rp.pClearValues = clears.data();
    vkCmdBeginRenderPass(c, &rp, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vpt{0, 0, (float)width, (float)height, 0, 1};
    VkRect2D sc{{0, 0}, {(uint32_t)width, (uint32_t)height}};
    vkCmdSetViewport(c, 0, 1, &vpt); vkCmdSetScissor(c, 0, 1, &sc);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, 1, &descSets[frame], 0, nullptr);

    // Floor.
    glm::mat4 ident(1);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, floorPipe);
    vkCmdPushConstants(c, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &ident);
    VkDeviceSize off0 = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &floorVbuf, &off0);
    vkCmdDraw(c, 6, 1, 0, 0);

    // Character.
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipe);
    vkCmdPushConstants(c, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &m);
    vkCmdBindVertexBuffers(c, 0, 1, &vbuf, &off0);
    vkCmdBindIndexBuffer(c, ibuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(c, indexCount, 1, 0, 0, 0);

    vkCmdEndRenderPass(c);
    vkEndCommandBuffer(c);

    VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1; si.pWaitSemaphores = &imgAvail[frame]; si.pWaitDstStageMask = &wait;
    si.commandBufferCount = 1; si.pCommandBuffers = &c;
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &renderDone[frame];
    VK_CHECK(vkQueueSubmit(gfxQueue, 1, &si, inFlight[frame]));

    VkPresentInfoKHR pr{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pr.waitSemaphoreCount = 1; pr.pWaitSemaphores = &renderDone[frame];
    pr.swapchainCount = 1; pr.pSwapchains = &swap.swapchain; pr.pImageIndices = &imgIdx;
    VkResult pres = vkQueuePresentKHR(presentQueue, &pr);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) recreateSwapchain();
  }

  void recreateSwapchain() {
    vkDeviceWaitIdle(dev);
    for (auto fb : framebuffers) vkDestroyFramebuffer(dev, fb, nullptr);
    for (auto v : swapViews) vkDestroyImageView(dev, v, nullptr);
    vkb::destroy_swapchain(swap);
    vkDestroyImageView(dev, depthView, nullptr);
    vmaDestroyImage(allocator, depthImage, depthAlloc);
    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    while (w == 0 || h == 0) { glfwGetFramebufferSize(window, &w, &h); glfwWaitEvents(); }
    width = w; height = h;
    createSwapchain(); createDepth(); createFramebuffers();
  }

  void mainLoop() {
    uint32_t frame = 0;
    double prev = glfwGetTime();
    float anim = 0.0f;
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
      double now = glfwGetTime();
      float dt = float(now - prev); prev = now;
      anim += dt;
      updateCharacter(dt);
      recordAndDraw(frame, anim);
      frame = (frame + 1) % kMaxFrames;
    }
    vkDeviceWaitIdle(dev);
  }

  void cleanup() {
    for (int i = 0; i < kMaxFrames; ++i) {
      vkDestroySemaphore(dev, imgAvail[i], nullptr);
      vkDestroySemaphore(dev, renderDone[i], nullptr);
      vkDestroyFence(dev, inFlight[i], nullptr);
      vmaDestroyBuffer(allocator, camBuf[i], camAlloc[i]);
      vmaDestroyBuffer(allocator, jointBuf[i], jointAlloc[i]);
    }
    vmaDestroyBuffer(allocator, vbuf, valloc);
    vmaDestroyBuffer(allocator, ibuf, ialloc);
    vmaDestroyBuffer(allocator, floorVbuf, floorAlloc);
    vkDestroyDescriptorPool(dev, descPool, nullptr);
    vkDestroyDescriptorSetLayout(dev, setLayout, nullptr);
    vkDestroyPipeline(dev, meshPipe, nullptr);
    vkDestroyPipeline(dev, floorPipe, nullptr);
    vkDestroyPipelineLayout(dev, pipeLayout, nullptr);
    for (auto fb : framebuffers) vkDestroyFramebuffer(dev, fb, nullptr);
    vkDestroyRenderPass(dev, renderPass, nullptr);
    vkDestroyImageView(dev, depthView, nullptr);
    vmaDestroyImage(allocator, depthImage, depthAlloc);
    for (auto v : swapViews) vkDestroyImageView(dev, v, nullptr);
    vkb::destroy_swapchain(swap);
    vmaDestroyAllocator(allocator);
    vkDestroyCommandPool(dev, cmdPool, nullptr);
    vkb::destroy_device(vkbDev);
    vkDestroySurfaceKHR(vkbInst.instance, surface, nullptr);
    vkb::destroy_instance(vkbInst);
    glfwDestroyWindow(window);
    glfwTerminate();
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string glb = "rock_char.glb";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--glb") == 0 && i + 1 < argc) glb = argv[++i];
  }
  try {
    App app;
    app.run(glb);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
