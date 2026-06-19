# Apps depend on the CUDA render/runtime path, so this subdirectory is only added when
# NCG_WITH_CUDA (see top-level CMakeLists.txt).
add_library(ncg_app_common INTERFACE)
target_include_directories(ncg_app_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(ncg_app_common INTERFACE ncg::core)

add_executable(ncg_viewer ncg_viewer/main.cpp)
target_link_libraries(ncg_viewer
  PRIVATE ncg_app_common ncg::body ncg::recon ncg::runtime ncg::io ncg::record ncg::warnings)

add_executable(ncg_cli ncg_cli/main.cpp)
target_link_libraries(ncg_cli
  PRIVATE ncg_app_common ncg::body ncg::recon ncg::runtime ncg::fit ncg::io ncg::select
          ncg::record ncg::warnings)
