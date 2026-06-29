# ncg_add_test(<name>
#     SOURCES <.cpp ...>
#     DEPS    <targets ...>
#     LABELS  <ctest labels ...>)   # e.g. cuda, golden
#
# Builds a Catch2 test executable and auto-registers each TEST_CASE with CTest.
include_guard(GLOBAL)
include(Catch)   # provided by Catch2's CMake package (catch_discover_tests)

function(ncg_add_test name)
  cmake_parse_arguments(A "" "" "SOURCES;DEPS;LABELS" ${ARGN})
  add_executable(${name} ${A_SOURCES})
  target_link_libraries(${name}
    PRIVATE Catch2::Catch2WithMain ${A_DEPS} ncg::warnings)
  target_include_directories(${name}
    PRIVATE ${CMAKE_SOURCE_DIR}/tests/support)
  # Pass the repo root so tests can locate data/golden/... deterministically.
  target_compile_definitions(${name} PRIVATE NCG_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  # Escape semicolons so a multi-label value (e.g. "cuda;nerf") is passed to
  # set_tests_properties as ONE list-valued LABELS property instead of splitting the
  # PROPERTIES argument list (which silently drops every label after the first).
  string(REPLACE ";" "\\;" _ncg_labels "${A_LABELS}")
  # A Catch2 case that SKIP()s (e.g. a CUDA/asset-gated test on a box without the asset) exits 4 and
  # prints "SKIPPED:". Match that in the test OUTPUT so CTest records it as Skipped, not Failed —
  # output-based and exit-code-independent, so it behaves identically on the H100 and a Mac CPU build.
  #
  # SKIP_REGULAR_EXPRESSION MUST come before LABELS: a multi-label value (e.g. "cuda;diffuse") can be
  # written space-separated into the generated set_tests_properties() call, which would shift every
  # following key/value pair by one and corrupt the SKIP property. Leading it keeps the skip intact;
  # at worst a trailing label is dropped (cosmetic), never the skip behaviour.
  catch_discover_tests(${name}
    PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED:" LABELS "${_ncg_labels}")
endfunction()
