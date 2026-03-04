set(SOURCES
    high_order_collision.cpp
    high_order_collision.hpp
    high_order_collision_3d.cpp
    high_order_collision_3d.hpp
    triple_pair_collision.cpp
    triple_pair_collision.hpp
    high_order_collision_dict.cpp
    high_order_collision_dict.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(ipc_toolkit PRIVATE ${SOURCES})
