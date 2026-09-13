add_library(image_container STATIC
	src/image_container.c
	include/image_container.h
    include/image.hpp
)

set_target_properties(image_container PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(image_container PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(image_container PUBLIC
    ${OS_FRONTIER_LIBS}
)
