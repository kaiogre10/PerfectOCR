cmake_minimum_required(VERSION 3.23)

add_library(workers
    src/image_loader.cpp
    src/angle_corrector.cpp
	src/poly_gone.cpp
)

set_target_properties(workers PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(workers PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/components/buffer_container/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/components/file_handler/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/components/c_utils/include>
    "${OPENCV_INCLUDE_DIR}"
)

target_link_directories(workers PRIVATE
    "${OPENCV_LIB_PATH}"
)

target_link_libraries(workers PRIVATE
    file_handler
    c_utils
    buffer_container
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    ${OS_FRONTIER_LIBS}
)
