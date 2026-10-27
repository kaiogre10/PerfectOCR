# ─── Cython: compiled_utils.pyx → compiled_utils.cpp ───
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/compiled_utils.cpp"
    COMMAND ${Python3_EXECUTABLE} -m cython --cplus -3
            -I "${CMAKE_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/compiled_utils.pyx"
            -o "${CMAKE_CURRENT_SOURCE_DIR}/compiled_utils.cpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/compiled_utils.pyx"
            "${CMAKE_CURRENT_SOURCE_DIR}/compiled_utils.pxd"
    COMMENT "Cython: compiled_utils.pyx → compiled_utils.cpp"
    VERBATIM
)
add_library(compiled_utils MODULE "${CMAKE_CURRENT_SOURCE_DIR}/compiled_utils.cpp")
target_link_libraries(compiled_utils PRIVATE Python3::Module)
configure_compiled_target(compiled_utils)