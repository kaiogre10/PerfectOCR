# ─── Cython: image.pyx → image.cpp ───
add_custom_command(
    OUTPUT  "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp"
    COMMAND ${Python3_EXECUTABLE} -m cython
            --cplus -3
            -I "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/image.pyx"
            -o "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/image.pyx"
            "${CMAKE_CURRENT_SOURCE_DIR}/image.pxd"
            "${CMAKE_CURRENT_SOURCE_DIR}/image_loader.pxd"
    COMMENT "Cython: image.pyx → image.cpp"
    VERBATIM
)

add_library(image MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp"
    "${CMAKE_SOURCE_DIR}/components/workers/image_preparation/src/image_loader.cpp"
    "${CMAKE_SOURCE_DIR}/components/workers/image_preparation/src/angle_corrector.cpp"
)

target_include_directories(image PRIVATE
    "${CMAKE_SOURCE_DIR}/components/workers/image_preparation/include"
    "${CMAKE_SOURCE_DIR}/components/image_container/include"
    "${CMAKE_SOURCE_DIR}/components/file_handler/include"
    "${CMAKE_SOURCE_DIR}/components/c_utils/include"
    "${CMAKE_SOURCE_DIR}/third_party/opencv_dep/install/include/opencv4"
)

target_link_libraries(image PRIVATE
    Python3::Module
    image_container
    file_handler
    c_utils
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)

configure_compiled_target(image)