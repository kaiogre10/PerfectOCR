# ─── Cython: image.pyx → image.cpp ───
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp"
    COMMAND ${Python3_EXECUTABLE} -m cython --cplus -3
            -I "${CMAKE_SOURCE_DIR}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../container"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../compiled_utils"
            "${CMAKE_CURRENT_SOURCE_DIR}/image.pyx"
            -o "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/image.pyx"
            "${CMAKE_CURRENT_SOURCE_DIR}/../container/container.pxd"
            "${CMAKE_CURRENT_SOURCE_DIR}/../container/workers.pxd"
            "${CMAKE_CURRENT_SOURCE_DIR}/../compiled_utils/compiled_utils.pxd"
    COMMENT "Cython: image.pyx → image.cpp"
    VERBATIM
)

add_library(image MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp"
    "${CMAKE_SOURCE_DIR}/components/workers/src/image_loader.cpp"
    "${CMAKE_SOURCE_DIR}/components/workers/src/angle_corrector.cpp"
)

target_include_directories(image PRIVATE
    "${CMAKE_SOURCE_DIR}/components/buffer_container/include"
    "${CMAKE_SOURCE_DIR}/components/workers/include"
    "${CMAKE_SOURCE_DIR}/components/file_handler/include"
    "${CMAKE_SOURCE_DIR}/components/c_utils/include"
    "${CMAKE_SOURCE_DIR}/third_party/opencv_dep/install/include/opencv4"
)

target_link_libraries(image PRIVATE
    Python3::Module
    container
    file_handler
    c_utils
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)

configure_compiled_target(image)