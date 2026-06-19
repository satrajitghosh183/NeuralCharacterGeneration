#include <ncg/mesh/extract.hpp>

#include <ncg/core/error.hpp>

namespace ncg::mesh {

TriMesh extract_mesh(const recon::GaussianCloud& /*gaussians*/, int /*grid_resolution*/) {
  NCG_NOT_IMPLEMENTED();
}

}  // namespace ncg::mesh
