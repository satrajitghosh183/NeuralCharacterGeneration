# SMPL-X LBS unit tests (CPU). nlohmann_json for golden_fixture.hpp.
ncg_add_test(test_smplx_lbs
  SOURCES test_smplx_lbs.cpp
  DEPS    ncg::body ncg::io nlohmann_json::nlohmann_json
  LABELS  "body")
