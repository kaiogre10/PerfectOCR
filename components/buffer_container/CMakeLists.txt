add_library(buffer_container STATIC
	src/buffer_container.c
	src/buffer_container.cpp
	include/buffer_container.h
	include/buffer_container.hpp
)

set_target_properties(buffer_container PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(buffer_container PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(buffer_container PUBLIC
    ${OS_FRONTIER_LIBS}
)