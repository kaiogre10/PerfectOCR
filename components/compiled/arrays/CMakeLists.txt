# ─── Cython: arrays.pyx → arrays.cpp ───
add_custom_command(
    OUTPUT  "${CMAKE_CURRENT_SOURCE_DIR}/arrays.cpp"
    COMMAND ${Python3_EXECUTABLE} -m cython
            --cplus -3
            -I "${CMAKE_CURRENT_SOURCE_DIR}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../image"
            "${CMAKE_CURRENT_SOURCE_DIR}/arrays.pyx"
            -o "${CMAKE_CURRENT_SOURCE_DIR}/arrays.cpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/arrays.pyx"
            "${CMAKE_CURRENT_SOURCE_DIR}/../image/image.pxd"
    COMMENT "Cython: arrays.pyx → arrays.cpp"
    VERBATIM
)

add_library(arrays MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/arrays.cpp"
)

target_link_libraries(arrays PRIVATE
    Python3::Module
    image_container
)

configure_compiled_target(arrays)