set(SOURCES
    high_order_collision.cpp
    high_order_collision.hpp
    triple_pair_collision.cpp
    triple_pair_collision.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(ipc_toolkit PRIVATE ${SOURCES})
