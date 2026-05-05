set(SOURCES
  adaptive_support.cpp
  adaptive_support.hpp
  high_order_collisions.cpp
  high_order_collisions.hpp
  high_order_collisions_builder.cpp
  high_order_collisions_builder.hpp
  high_order_contact_potential.hpp
  high_order_contact_potential.cpp
  quadrature_potential.cpp
  quadrature_potential.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(ipc_toolkit PRIVATE ${SOURCES})

################################################################################
# Subfolders
################################################################################

add_subdirectory(collisions)
