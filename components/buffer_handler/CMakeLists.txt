add_library(buffer_handler SHARED
    "buffer_handler.cpp"
    "buffer_handler.h"
)

# Ruta de inclusión para encontrar "containers.h" de forma limpia
target_include_directories(buffer_handler 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# VINCULACIÓN CRÍTICA: Resuelve el LNK2019 inyectando containers.lib en buffer_handler.dll
target_link_libraries(buffer_handler 
    PRIVATE 
        containers
        ${OS_FRONTIER_LIBS}
)