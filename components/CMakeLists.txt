# components/CMakeLists
# 1. Dependencias Internas (Compilación STATIC)
cmake_minimum_required(VERSION 3.23)
project(components LANGUAGES C CXX)

set(OPENCV_LIB_PATH "${LIBS_DIR}/opencv_dep")

if (WIN32)
    set(XTENSOR_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/xtensor_dep/install/include")
    set(OPENCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/opencv_dep/install/include")
else()
    set(XTENSOR_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/xtensor_dep/install/include")
    set(OPENCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/opencv_dep/install/include/opencv4")
endif()

add_subdirectory(c_utils)
add_subdirectory(file_handler)
add_subdirectory(image_container)
add_subdirectory(payload_container)

# 2. Interfaces de Exportación (Compilación SHARED)
add_subdirectory(workers)
add_subdirectory(buffer_handler)

## 3. Bindings Python (Cython)
add_subdirectory(compiled)