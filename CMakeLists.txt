# PerfectOCR/CMakeLists.txt
cmake_minimum_required(VERSION 3.23)
project(PerfectOCR LANGUAGES C CXX)

if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

get_filename_component(_root_name "${CMAKE_SOURCE_DIR}" NAME)

if(NOT ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}" AND "${_root_name}" STREQUAL "${PROJECT_NAME}"))
    message(FATAL_ERROR "ERROR: SOURCE DIR, CURRENT SOURCE DIR Y PROJECT_NAME NO SON IGUALES ENTRE SI, REVISAR CACHE Y/O ESTABLECER EL ROOT DEL PROYECTO A LA CARPETA DE ESTE SCRIPT")
endif()

include(cmake/GlobalCompiler.cmake)
# 1. Definición de la ruta de destino para los componentes
get_filename_component(LIBRARIES_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libraries" ABSOLUTE)

if(NOT EXISTS "${LIBRARIES_OUTPUT_DIR}")
    file(MAKE_DIRECTORY "${LIBRARIES_OUTPUT_DIR}")
endif()

set(LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs")
set(OPENCV_LIB_PATH "${LIBS_DIR}/opencv_dep")

if (WIN32)
    set(OPENCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/opencv_dep/install/include")
else ()
    set(OPENCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/opencv_dep/install/include/opencv4")
endif()

# Desactivar subcarpetas por configuración en generadores multi-configuración
foreach(OUTPUT_TYPE RUNTIME LIBRARY ARCHIVE)
    set(CMAKE_${OUTPUT_TYPE}_OUTPUT_DIRECTORY "${LIBRARIES_OUTPUT_DIR}")
    foreach(CONFIG DEBUG RELEASE RELWITHDEBINFO)
        set(CMAKE_${OUTPUT_TYPE}_OUTPUT_DIRECTORY_${CONFIG} "${LIBRARIES_OUTPUT_DIR}")
    endforeach()
endforeach()

# Remoción de prefijos para carga limpia
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_PREFIX "")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_subdirectory(components)
add_subdirectory(bidings)
