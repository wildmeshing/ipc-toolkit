set(SOURCES
  offset_collisions.cpp
  offset_collisions.hpp
  offset_collisions_builder.cpp
  offset_collisions_builder.hpp
  offset_contact_potential.hpp
  offset_contact_potential.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(ipc_toolkit PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(collisions)
