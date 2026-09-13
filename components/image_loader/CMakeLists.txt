add_library(image_loader STATIC
    src/image_loader.cpp
)

set_target_properties(image_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(image_loader PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/image_container/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/file_handler/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/c_utils/include>
    "${OPENCV_INCLUDE_DIR}"
)

target_link_directories(image_loader PRIVATE
    "${LIBS_DIR}"
)

target_link_libraries(image_loader PRIVATE
    file_handler
    c_utils
    image_container
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    ${OS_FRONTIER_LIBS}
)