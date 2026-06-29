include(NcgAddTest)

add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(body)
add_subdirectory(recon)
add_subdirectory(geom)
add_subdirectory(diffuse)
add_subdirectory(select)
add_subdirectory(record)
add_subdirectory(mesh)
add_subdirectory(material)
add_subdirectory(rig)
# runtime/fit/nerf carry the CUDA renderer (their libs are added only under NCG_WITH_CUDA at the
# top level); their tests link those libs, so gate them identically. On a host-only (Mac CPU) build
# the CPU-clean subset above still builds + tests, so local development continues without an H100.
if(NCG_WITH_CUDA)
  add_subdirectory(runtime)
  add_subdirectory(fit)
  add_subdirectory(nerf)
endif()
add_subdirectory(golden)
