# ─── Cython: container.pyx → container.cpp ───
add_custom_command(
    OUTPUT  "${CMAKE_CURRENT_SOURCE_DIR}/container.cpp"
    COMMAND ${Python3_EXECUTABLE} -m cython
            --cplus -3
            -I "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/container.pyx"
            -o "${CMAKE_CURRENT_SOURCE_DIR}/container.cpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/container.pyx"
            "${CMAKE_SOURCE_DIR}/bidings/compiled_utils/compiled_utils.pxd"
            "${CMAKE_CURRENT_SOURCE_DIR}/container.pxd"
    COMMENT "Cython: container.pyx → container.cpp"
    VERBATIM
)

add_library(container MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/container.cpp"
)

target_include_directories(container PRIVATE
    "${CMAKE_SOURCE_DIR}/components/buffer_container/include"
)

target_link_libraries(container PRIVATE
    Python3::Module
    buffer_container
)

configure_compiled_target(container)