add_library(containers SHARED
	"image.hpp"
    "containers.h" 
    "containers.cpp"
	"image_container.c"
	"image_container.h"
)

target_include_directories(containers 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(containers 
    PUBLIC 
        ${OS_FRONTIER_LIBS}
)