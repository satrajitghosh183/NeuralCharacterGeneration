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

/// Information-greedy view selection (leg #3, bad-upload handling): from M candidate views with
/// per-vertex coverage weights (e.g. visibility from each photo), greedily pick `k` that maximize
/// the total effective coverage  sum_v log(1 + sum_{i in S} coverage[i,v]).  The objective is
/// monotone submodular, so greedy is within (1 - 1/e) of optimal; it rewards views that see
/// *new / under-covered* surface and naturally starves redundant or empty (junk) uploads.
///   coverage : [M,V] non-negative per-vertex weight for each candidate view
/// Returns the selected view indices (size = min(k, M)).
std::vector<int64_t> select_views(const Tensor& coverage, int64_t k);

}  // namespace ncg::select
