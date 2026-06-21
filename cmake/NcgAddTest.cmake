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
  catch_discover_tests(${name}
    PROPERTIES LABELS "${_ncg_labels}")
endfunction()
