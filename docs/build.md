# Building NeuralCharGen (H100 Linux box)

Source of truth for build/run commands and toolchain version pins. The dev loop is:
**author on Mac → `git push` → build & run here on the H100 → paste errors back.**

## Toolchain (pin these; mismatches cause link/ABI failures)
| Component | Required | Notes |
|---|---|---|
| GPU | NVIDIA H100 (Hopper, sm_90) | build targets `90-real` only |
| CUDA Toolkit | 12.x | `nvcc --version` |
| Driver | matching CUDA 12.x | `nvidia-smi` |
| LibTorch | CUDA 12.x build, **cxx11-ABI** | download the cxx11-ABI zip, not pre-cxx11 |
| Host compiler | GCC matching the LibTorch build (GCC 11–13) | ABI must match LibTorch |
| CMake | ≥ 3.27 | presets v6, good CUDA arch support |
| Ninja, ccache | any recent | fast incremental rebuilds |
| git-lfs | any | for `data/` weights + golden tensors |

> The cxx11-ABI flag is propagated automatically by `find_package(Torch)`. **Never** set
> `_GLIBCXX_USE_CXX11_ABI` by hand — match the GCC to the LibTorch build instead.

## One-time setup
```bash
git lfs install
export LIBTORCH_ROOT=/abs/path/to/libtorch     # add to ~/.bashrc
# (optional) machine-specific overrides without editing tracked presets:
#   create CMakeUserPresets.json (gitignored) inheriting h100-release
```

## Configure / build / test
```bash
cmake --preset h100-release          # configure (Ninja, ccache, sm_90)
cmake --build --preset h100-release  # build everything
ctest --preset h100                  # run all tests, output-on-failure
```
Useful subsets / tight loops:
```bash
ctest --preset h100 -L cuda          # custom-kernel-vs-LibTorch tests
ctest --preset h100 -L golden        # weight-port parity tests
ctest --preset h100 -R saxpy         # by name regex
cmake --build --preset h100-release --target test_elementwise_kernel   # single target
```

## First-build smoke test (validates the whole toolchain)
```bash
cmake --preset h100-release && cmake --build --preset h100-release
ctest --preset h100 -L cuda          # expect: test_elementwise_kernel PASS
```
If this is green, CUDA + LibTorch + CMake + Catch2 are correctly wired and we can proceed.

## Debugging CUDA faults (the build-blind multiplier)
```bash
# Pinpoint illegal-memory/launch faults to an exact kernel + access:
compute-sanitizer --tool memcheck ./build/h100-release/tests/core/test_elementwise_kernel
# Force per-launch sync so async faults surface at the launch site:
NCG_CUDA_SYNC=1 ctest --preset h100 -R <name>
# Verbose logs:
NCG_LOG_LEVEL=debug ./build/h100-release/...
```
Paste the **full** first error (file:line + kernel name) back — that's what localizes the fix.

## Common pitfalls
- **`find_package(Torch)` not found** → `CMAKE_PREFIX_PATH`/`LIBTORCH_ROOT` not set or points at the wrong unzip dir (must contain `share/cmake/Torch`).
- **Undefined symbols / std::string ABI errors at link** → pre-cxx11 LibTorch or mismatched GCC. Use the cxx11-ABI LibTorch.
- **fp32 parity test fails by a hair on Hopper** → TF32. Parity code calls `ncg::set_deterministic_fp32(true)`; ensure it runs before the model.
