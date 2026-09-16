# third_party/CMakelists
cmake_minimum_required(VERSION 3.20)

# Aplicar config ANTES de project()
#include("${CMAKE_CURRENT_SOURCE_DIR}/opencv_dep/opencv_config.cmake")

add_subdirectory(opencv_dep)