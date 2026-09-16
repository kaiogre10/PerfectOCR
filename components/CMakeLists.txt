# components/CMakeLists
# 1. Dependencias Internas (Compilación STATIC)
cmake_minimum_required(VERSION 3.20)
add_subdirectory(c_utils)
add_subdirectory(file_handler)
add_subdirectory(image_container)
add_subdirectory(payload_container)

# 2. Interfaces de Exportación (Compilación SHARED)
add_subdirectory(workers)
add_subdirectory(buffer_handler)

# 3. Bindings Python (Cython)
add_subdirectory(compiled)