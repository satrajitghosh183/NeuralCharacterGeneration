# Recorder + metrics tests (CPU).
ncg_add_test(test_record
  SOURCES test_record.cpp
  DEPS    ncg::record
  LABELS  "record")
