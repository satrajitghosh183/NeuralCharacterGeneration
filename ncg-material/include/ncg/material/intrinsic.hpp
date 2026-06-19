#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg::material {

/// PBR material maps for relighting (Phase 4). albedo/roughness/normal as CHW tensors.
struct PbrMaps {
  Tensor albedo;
  Tensor roughness;
  Tensor normal;
};

/// Intrinsic decomposition / delighting (IDArb-class). SKELETON — Phase 4 (docs/plan.md).
PbrMaps decompose(const Tensor& image_chw);

}  // namespace ncg::material
