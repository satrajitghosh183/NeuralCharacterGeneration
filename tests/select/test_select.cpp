#include <catch2/catch_test_macros.hpp>

#include <ncg/select/selector.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <vector>

// 12 candidate views: 0-3 cover four disjoint surface bands (together = full coverage),
// 4-7 are redundant duplicates of band 0, 8-11 are near-empty "junk" uploads. A good selector
// picking 4 should choose the four diverse views (full coverage) and never the junk.
TEST_CASE("select_views picks diverse coverage, rejects junk", "[select]") {
  const int M = 12;
  const int V = 400;
  auto cov = torch::zeros({M, V});
  for (int b = 0; b < 4; ++b) cov[b].narrow(0, b * 100, 100).fill_(1.0);  // disjoint bands
  for (int r = 4; r < 8; ++r) cov[r].narrow(0, 0, 100).fill_(1.0);        // redundant w/ view 0
  for (int j = 8; j < 12; ++j) cov[j].fill_(0.01);                        // junk: ~no coverage

  const auto sel = ncg::select::select_views(cov, 4);
  REQUIRE(sel.size() == 4);

  // No junk view selected.
  for (int64_t idx : sel) REQUIRE(idx < 8);

  // The selected views together cover every vertex (min summed coverage > 0).
  auto acc = torch::zeros({V});
  for (int64_t idx : sel) acc = acc + cov[idx];
  REQUIRE(acc.min().item<double>() > 0.5);  // full coverage — only achievable via the 4 bands
}

TEST_CASE("select_views beats a fixed redundant subset on coverage", "[select]") {
  const int M = 12;
  const int V = 400;
  auto cov = torch::zeros({M, V});
  for (int b = 0; b < 4; ++b) cov[b].narrow(0, b * 100, 100).fill_(1.0);
  for (int r = 4; r < 8; ++r) cov[r].narrow(0, 0, 100).fill_(1.0);
  for (int j = 8; j < 12; ++j) cov[j].fill_(0.01);

  const auto sel = ncg::select::select_views(cov, 4);
  auto greedy = torch::zeros({V});
  for (int64_t idx : sel) greedy = greedy + cov[idx];
  // A redundant subset {0,4,5,6} leaves 3/4 of the surface unseen.
  auto redundant = cov[0] + cov[4] + cov[5] + cov[6];
  REQUIRE((greedy > 0).sum().item<int64_t>() > (redundant > 0).sum().item<int64_t>());
}
