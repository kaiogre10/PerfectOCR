add_library(c_utils SHARED
	"c_utils.cpp"
	"c_utils.hpp"
)

target_include_directories(c_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(c_utils
    PUBLIC
        ${OS_FRONTIER_LIBS}
)