# Manteniendo la declaración exacta de tu componente original, sin alterar enlazados ficticios
add_library(image_preparation
    src/image_loader.cpp
#    src/angle_corrector.cpp
)

set_target_properties(image_preparation PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(image_preparation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/components/image_container/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/components/file_handler/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/components/c_utils/include>
    "${OPENCV_INCLUDE_DIR}"
)

target_link_directories(image_preparation PRIVATE
    "${LIBS_DIR}"
)

target_link_libraries(image_preparation PRIVATE
    file_handler
    c_utils
    image_container
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    ${OS_FRONTIER_LIBS}
)
