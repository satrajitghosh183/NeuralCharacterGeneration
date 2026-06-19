#include <ncg/rig/rig.hpp>

#include <ncg/core/error.hpp>

namespace ncg::rig {

RiggedMesh autorig(const Tensor& /*vertices*/, const Tensor& /*faces*/) { NCG_NOT_IMPLEMENTED(); }

void export_rigged(const RiggedMesh& /*mesh*/, const std::string& /*path*/) {
  NCG_NOT_IMPLEMENTED();
}

}  // namespace ncg::rig
