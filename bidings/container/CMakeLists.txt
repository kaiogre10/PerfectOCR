# ─── Cython: container.pyx → container.cpp ───
add_library(container INTERFACE)

target_include_directories(container INTERFACE
    "${CMAKE_SOURCE_DIR}/components/buffer_container/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../compiled_utils"
)

target_link_libraries(container INTERFACE
    buffer_container
    compiled_utils
)