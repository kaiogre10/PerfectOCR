# components/CMakeLists
# 1. Dependencias Internas (Compilación STATIC)
cmake_minimum_required(VERSION 3.23)
project(components LANGUAGES C CXX)

add_subdirectory(c_utils)
add_subdirectory(file_handler)
add_subdirectory(buffer_container)
add_subdirectory(payload_container)

# 2. Interfaces de Exportación (Compilación SHARED)
add_subdirectory(workers)
add_subdirectory(buffer_handler)