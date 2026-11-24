# components/c_utils/CMakeLists
cmake_minimum_required(VERSION 3.25)
add_library(c_utils STATIC
	src/c_utils.cpp
)

set_target_properties(c_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(c_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    "${OPENCV_INCLUDE_DIR}"
)

target_link_directories(c_utils PRIVATE
    "${OPENCV_LIB_PATH}"
)

target_link_libraries(c_utils PRIVATE
    opencv_core
    opencv_imgproc
    ${OS_FRONTIER_LIBS}
)
