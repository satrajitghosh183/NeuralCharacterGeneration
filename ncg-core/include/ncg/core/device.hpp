#pragma once

#include <torch/torch.h>

namespace ncg {

/// True if a CUDA device is available at runtime.
bool cuda_available();

/// True if Apple-Silicon MPS (Metal) is available at runtime.
bool mps_available();

/// CUDA device 0 if available, else CPU. The pipeline's default compute device.
at::Device default_device();

/// Best GPU-class device for tensor compute: CUDA > MPS > CPU. Env override NCG_DEVICE=cuda|mps|cpu.
/// True when compute_device() is a GPU of some kind (CUDA or MPS).
at::Device compute_device();
bool gpu_available();

/// Disable TF32 on cuBLAS/cuDNN so fp32 parity tests are reproducible on Hopper
/// (TF32 is on by default and silently exceeds fp32 tolerances). Call before parity runs.
void set_deterministic_fp32(bool enable);

}  // namespace ncg
