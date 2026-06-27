# Phase B: Δv out-of-subspace geometry solver + observability gate, synthetic GT — CPU.
ncg_add_test(test_solve_geometry
  SOURCES test_solve_geometry.cpp
  DEPS    ncg::geom
  LABELS  "geom")
