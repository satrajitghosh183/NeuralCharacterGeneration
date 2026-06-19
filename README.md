# NeuralCharGen

Casual personal photos → a **hyperrealistic, rigged, relightable, full-body 3D avatar** that drops into Unity and Unreal. Built entirely in C++/CUDA.

- **Architecture & roadmap:** [`docs/plan.md`](docs/plan.md)
- **SOTA research / model choices:** [`docs/research-findings.md`](docs/research-findings.md)
- **Build instructions (H100):** [`docs/build.md`](docs/build.md)
- **Contributor guide / conventions:** [`CLAUDE.md`](CLAUDE.md)

## Quick start (H100)
```bash
export LIBTORCH_ROOT=/path/to/libtorch
cmake --preset h100-release
cmake --build --preset h100-release
ctest --preset h100 -L cuda     # toolchain smoke test
```

Status: Phase 0 (foundations). See `docs/plan.md` for the phased plan.
