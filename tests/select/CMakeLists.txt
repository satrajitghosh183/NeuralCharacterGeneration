# View selection (information-greedy) — CPU.
ncg_add_test(test_select
  SOURCES test_select.cpp
  DEPS    ncg::select
  LABELS  "select")

# Phase A: robust subject gate / contamination rejection (C2) on synthetic embeddings — CPU.
ncg_add_test(test_subject_gate
  SOURCES test_subject_gate.cpp
  DEPS    ncg::select
  LABELS  "select")
