# ncg_add_library(<name>
#     SOURCES       <.cpp ...>
#     CUDA_SOURCES  <.cu ...>      # only compiled when NCG_WITH_CUDA
#     PUBLIC_DEPS   <targets ...>
#     PRIVATE_DEPS  <targets ...>)
#
# Creates target <name> + alias ncg::<name>, with public headers under include/.
# DRY wrapper so every module's CMakeLists stays a few lines.
include_guard(GLOBAL)

function(ncg_add_library name)
  cmake_parse_arguments(A "" "" "SOURCES;CUDA_SOURCES;PUBLIC_DEPS;PRIVATE_DEPS" ${ARGN})

  set(_sources ${A_SOURCES})
  if(NCG_WITH_CUDA)
    list(APPEND _sources ${A_CUDA_SOURCES})
  elseif(A_CUDA_SOURCES)
    message(STATUS "NCG: ${name}: skipping CUDA sources (host-only configure).")
  endif()

  add_library(${name} ${_sources})
  add_library(ncg::${name} ALIAS ${name})

  target_include_directories(${name}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  target_link_libraries(${name}
    PUBLIC  ${A_PUBLIC_DEPS}
    PRIVATE ${A_PRIVATE_DEPS} ncg::warnings)

  target_compile_features(${name} PUBLIC cxx_std_20)

  if(NCG_WITH_CUDA AND A_CUDA_SOURCES)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()
endfunction()
