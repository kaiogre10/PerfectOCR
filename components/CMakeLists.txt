cmake_minimum_required(VERSION 3.20)
project(components LANGUAGES C CXX)

if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BUILD_SHARED_LIBS "Fuerza la compilación de librerías dinámicas" ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "Intel OneAPI detectado. Activando optimización de arquitectura host.")
    
    if (WIN32)
        # Optimizaciones para icx-cl en Windows
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O3 /QxHost")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O3 /QxHost")
        
        # ELIMINACIÓN OBLIGATORIA: Purga de /Qm64 y flags obsoletos de Intel clásico
        foreach(flag_var 
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO
            CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_RELWITHDEBINFO
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

# 1. Definición de la ruta de destino global
set(GLOBAL_DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../bin")

# Configuraciones base para generadores mono-configuración (Ninja/Makefiles)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${GLOBAL_DESTINATION}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${GLOBAL_DESTINATION}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${GLOBAL_DESTINATION}")

# Desactivar subcarpetas por configuración en generadores multi-configuración (Visual Studio)
set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL "MinSizeRel") # Requerido por consistencia interna
foreach(OUTPUT_TYPE RUNTIME LIBRARY ARCHIVE)
    foreach(CONFIG DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        string(TOUPPER ${CONFIG} CONFIG_UPPER)
        set(CMAKE_${OUTPUT_TYPE}_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${GLOBAL_DESTINATION}")
    endforeach()
endforeach()

# Remoción de prefijos para carga limpia en ctypes
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_PREFIX "")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Orden secuencial por dependencias
add_subdirectory(buffer_handler)
add_subdirectory(containers)
add_subdirectory(image_loader)