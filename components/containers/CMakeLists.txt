add_library(containers SHARED 
    "containers.h" 
    "containers.cpp"
	"image_container.cpp"
	"image_container.hpp"
)

target_include_directories(containers 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(containers 
    PUBLIC 
        ${OS_FRONTIER_LIBS}
)