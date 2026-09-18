cmake_minimum_required(VERSION 3.20)
project(PerfectOCR LANGUAGES C CXX)

if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif ()

get_filename_component(_root_name "${CMAKE_SOURCE_DIR}" NAME)

if(NOT ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}" AND "${_root_name}" STREQUAL "${PROJECT_NAME}"))
    message(FATAL_ERROR "ERROR: SOURCE DIR, CURRENT SOURCE DIR Y PROJECT_NAME NO SON IGUALES ENTRE SI, REVISAR CACHE Y/O ESTABLECER EL ROOT DEL PROYECTO A LA CARPETA DE ESTE SCRIPT")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Fuerza la compilación de librerías dinámicas" ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "Intel OneAPI detectado. Activando optimización de arquitectura host.")
    
    if (WIN32)
        # Optimizaciones para icx-cl en Windows
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O3 /QxHost")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O3 /QxHost")
        
        # ELIMINACIÓN OBLIGATORIA: Purga de /Qm64 y flags obsoletos de Intel clásico
        foreach(flag_var 
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
            CMAKE_EXE_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS_DEBUG CMAKE_EXE_LINKER_FLAGS_RELEASE
            CMAKE_SHARED_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS_DEBUG CMAKE_SHARED_LINKER_FLAGS_RELEASE)
            if(DEFINED ${flag_var})
                string(REPLACE "/Qm64" "" ${flag_var} "${${flag_var}}")
                string(REPLACE "-qm64" "" ${flag_var} "${${flag_var}}")
                string(REPLACE "/Qnextgen" "" ${flag_var} "${${flag_var}}")
                string(REPLACE "-qnextgen" "" ${flag_var} "${${flag_var}}")
            endif()
        endforeach()
    else ()
        # Sintaxis GNU/Clang para Linux
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -xHost")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -xHost")
    endif ()
endif ()
# Preparamos una variable global con las librerías del kernel que requiere cada OS
if (UNIX AND NOT APPLE)
    # Linux exige estas librerías para la memoria compartida (shm_open) y hilos
    set(OS_FRONTIER_LIBS rt pthread)
else ()
    # Windows maneja la memoria proyectada nativamente sin librerías extra
    set(OS_FRONTIER_LIBS "")
endif ()

# 1. Definición de la ruta de destino para los componentes
set(LIBRARIES_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/libraries")

set(LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs")

if (WIN32)
    set(OPENCV_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv_dep/install/include/opencv2")
else()
    set(OPENCV_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv_dep/install/include/opencv4")
endif()

# Desactivar subcarpetas por configuración en generadores multi-configuración (Visual Studio)
foreach(OUTPUT_TYPE RUNTIME LIBRARY ARCHIVE)
    foreach(CONFIG DEBUG RELEASE)
        string(TOUPPER ${CONFIG} CONFIG_UPPER)
        set(CMAKE_${OUTPUT_TYPE}_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${LIBRARIES_OUTPUT_DIR}")
#/opt/clion-2026.1.3/bin/cmake/linux/x64/bin/cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=/opt/clion-2026.1.3/bin/ninja/linux/x64/ninja -G Ninja -S /home/kaiogre05/PerfectOCR -B /home/kaiogre05/PerfectOCR/cmake-build-release
    endforeach()
endforeach()

# Remoción de prefijos para carga limpia
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_PREFIX "")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_subdirectory(third_party)
add_subdirectory(components)