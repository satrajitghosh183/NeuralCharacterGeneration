#pragma once

#include <ncg/core/tensor.hpp>

#include <string>
#include <vector>

namespace ncg::select {

struct ScoredImage {
  std::string path;
  double sharpness;  // variance of the Laplacian (higher = sharper)
};

/// Variance-of-Laplacian sharpness of a CHW float image in [0,1]. A simple, fast, no-reference
/// proxy. Phase 1 placeholder for the learned quality-aware selector (legs #1/#3 in docs/plan.md).
double sharpness_score(const Tensor& image_chw);

/// Scores each path (descending by sharpness).
std::vector<ScoredImage> score_images(const std::vector<std::string>& paths);

/// Highest-sharpness path (or the first if the list is singletons). Throws if empty.
std::string select_best(const std::vector<std::string>& paths);

}  // namespace ncg::select
