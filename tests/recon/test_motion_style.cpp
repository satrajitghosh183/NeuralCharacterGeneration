#include <catch2/catch_test_macros.hpp>

#include <ncg/recon/motion_style.hpp>

#include <torch/torch.h>

#include <vector>

using ncg::recon::apply_motion_style;
using ncg::recon::MotionStyleConfig;
using ncg::recon::solve_motion_style;

// C4 (docs/method.md §13): a person's motion STYLE is a shared low-rank subspace; each ACTION
// exercises only part of it. Recovering the full style needs several DIFFERENT actions (the
// action-diversity analog of C1's lighting diversity), and the robust solver must beat naive
// pooling on cut/outlier frames (the C2 analog). Synthetic ground truth so error is measurable.
namespace {
constexpr int D = 60;       // pose-feature dim
constexpr int R = 6;        // style rank
constexpr int RA = 2;       // dims each action exercises
constexpr int T = 40;       // frames/clip

torch::Tensor ortho_rows(int r, int d) {  // [r,d] with orthonormal rows
  auto m = torch::randn({d, r});
  return std::get<0>(torch::linalg_qr(m)).t().contiguous();  // [r,d]
}
}  // namespace

TEST_CASE("motion style: action diversity recovers the style subspace (C4)", "[recon][motion]") {
  torch::manual_seed(0);
  const auto W_true = ortho_rows(R, D);  // [R,D] personal style subspace

  auto make = [&](int N, double rho) {
    std::vector<torch::Tensor> clips;
    for (int c = 0; c < N; ++c) {
      const auto Maction = torch::randn({RA, R});             // this action spans RA of R style dims
      const auto Phi = torch::matmul(torch::randn({T, RA}), Maction);  // [T,R] rank RA
      auto Y = torch::matmul(Phi, W_true) + 0.01 * torch::randn({T, D});
      if (rho > 0) {  // cut/outlier frames
        const auto bad = torch::rand({T}) < rho;
        Y = torch::where(bad.unsqueeze(1), torch::randn({T, D}) * 2.0, Y);
      }
      clips.push_back(Y);
    }
    return clips;
  };
  // Transfer: a held-out action that uses the FULL style subspace -> tests if all style dims were
  // recovered. Project its true poses onto the recovered style; error = subspace miss.
  auto transfer_err = [&](const torch::Tensor& W_rec) {
    const auto Ystar = torch::matmul(torch::randn({T, R}), W_true);  // full-rank content
    const auto proj = apply_motion_style(Ystar, W_rec);
    return (Ystar - proj).norm().item<double>() / Ystar.norm().clamp_min(1e-8).item<double>();
  };

  MotionStyleConfig cfg;
  cfg.rank = R;
  cfg.iterations = 60;
  const double e_div = transfer_err(solve_motion_style(make(6, 0.0), cfg).style);   // diverse
  const double e_few = transfer_err(solve_motion_style(make(2, 0.0), cfg).style);   // few actions
  INFO("transfer err: 6 actions=" << e_div << "  2 actions=" << e_few);
  REQUIRE(e_div < 0.10);   // diverse actions recover the style well
  REQUIRE(e_div < e_few);  // action diversity strictly helps (the C4 effect)
}

TEST_CASE("motion style: robust factorization beats naive on cut frames (C4)", "[recon][motion]") {
  torch::manual_seed(1);
  const auto W_true = ortho_rows(R, D);
  auto make = [&](int N, double rho) {
    std::vector<torch::Tensor> clips;
    for (int c = 0; c < N; ++c) {
      const auto Maction = torch::randn({RA, R});
      const auto Phi = torch::matmul(torch::randn({T, RA}), Maction);
      auto Y = torch::matmul(Phi, W_true) + 0.01 * torch::randn({T, D});
      const auto bad = torch::rand({T}) < rho;
      Y = torch::where(bad.unsqueeze(1), torch::randn({T, D}) * 2.0, Y);
      clips.push_back(Y);
    }
    return clips;
  };
  auto transfer_err = [&](const torch::Tensor& W_rec) {
    const auto Ystar = torch::matmul(torch::randn({T, R}), W_true);
    return (Ystar - apply_motion_style(Ystar, W_rec)).norm().item<double>() /
           Ystar.norm().clamp_min(1e-8).item<double>();
  };

  const auto clips = make(8, 0.30);  // 30% cut/outlier frames
  MotionStyleConfig rc;
  rc.rank = R;
  rc.iterations = 60;
  rc.robust = true;
  MotionStyleConfig pc = rc;
  pc.robust = false;
  const double er = transfer_err(solve_motion_style(clips, rc).style);
  const double ep = transfer_err(solve_motion_style(clips, pc).style);
  INFO("transfer err: robust=" << er << "  naive=" << ep);
  REQUIRE(er < ep);     // robustness helps under corruption
  REQUIRE(er < 0.15);   // and still recovers the style
}
