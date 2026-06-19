# Default to a single real H100 architecture. `90-real` emits SASS only (no PTX, no fat
# binary) which is the biggest single build-time win on a one-box target. Add `90-virtual`
# only if forward-compatible PTX is ever needed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 90-real CACHE STRING "CUDA architectures" FORCE)
endif()
