# ncg_game — "Rock Walker" (Vulkan)

A small native **Vulkan** game that loads the rigged character from `ncg_cli avatar`
(`rock_char.glb`), stands it on a grid floor, plays its skeletal animation, and lets you walk it
around with simple gravity/jump. No engine, no Python — pure C++/Vulkan, matching the project ethos.

## Where it runs
A machine **with a display and a Vulkan driver** — your **M4 Pro via MoltenVK**, or a Linux box with
a GPU. (Not the headless H100.) It's a separate CMake project so it never entangles the H100 build.

## Prerequisites (macOS / Apple Silicon)
```bash
brew install glfw glslang molten-vk vulkan-headers vulkan-loader cmake
# or install the LunarG Vulkan SDK (provides MoltenVK + glslangValidator + loader)
export VULKAN_SDK="$(brew --prefix molten-vk)"   # if not using the LunarG SDK
```
GLFW, vk-bootstrap, VMA, tinygltf and glm are fetched automatically by CMake.

## Build & run
```bash
cmake -S apps/ncg_game -B build/game -DCMAKE_BUILD_TYPE=Release
cmake --build build/game -j
./build/game/ncg_game --glb rock_char.glb
```

## Controls
| input | action |
|---|---|
| **W A S D** | walk the character (relative to camera) |
| **mouse drag** | orbit camera |
| **scroll** | zoom |
| **Space** | jump (gravity brings you back to the floor) |
| **Esc** | quit |

## Notes
- Renders the **mesh** representation (skinned, animated, vertex-colored). The high-fidelity
  Gaussian-splat look (`rock_char.ply`) is a follow-on: a splat render pass + the same skinning,
  driven by `rock_char.ply.skin` (see `docs/engine_character.md`).
- Skinning is LBS in `mesh.vert`; joint matrices are evaluated CPU-side each frame from the glTF
  animation (`jointMatrices()` in `main.cpp`) — the same math as `ncg::fit::deform_avatar`.
- Dev loop is the project standard: build on the target machine, paste errors back, iterate.
