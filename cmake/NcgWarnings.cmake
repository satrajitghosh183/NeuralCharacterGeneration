# Interface target carrying our warning policy. Link as PRIVATE on every ncg target.
# Kept strict but not -Werror (a remote build shouldn't die on a benign warning mid-iteration;
# CI can add -Werror later).
if(NOT TARGET ncg_warnings)
  add_library(ncg_warnings INTERFACE)
  add_library(ncg::warnings ALIAS ncg_warnings)

  set(_ncg_cxx_warnings
      -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
      -Wcast-align -Wunused -Woverloaded-virtual -Wconversion
      -Wsign-conversion -Wdouble-promotion -Wno-unknown-pragmas)

  target_compile_options(ncg_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${_ncg_cxx_warnings}>
    # For CUDA, forward host-side warnings to the host compiler only.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()
