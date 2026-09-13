cmake_minimum_required(VERSION 3.20)
project(third_party LANGUAGES C CXX)

if(WIN32)
    set(OS_FRONTIER_LIBS "")
else()
    set(OS_FRONTIER_LIBS rt pthread)
endif()

# ─── Output: todas las libs custom van a project_root/libs/ ───
set(LIBS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libs")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${LIBS_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIBS_OUTPUT_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${LIBS_OUTPUT_DIR}")

add_subdirectory(opencv_dep)