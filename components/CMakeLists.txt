cmake_minimum_required(VERSION 3.20)
project(components LANGUAGES CXX)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BUILD_SHARED_LIBS "Fuerza la compilación de librerías dinámicas" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "Intel OneAPI detectado. Activando optimización de arquitectura host.")
    # -xHost fuerza a que el compilador use las instrucciones AVX nativas de TU procesador
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -xHost")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -xHost")
endif()

# Preparamos una variable global con las librerías del kernel que requiere cada OS
if(UNIX AND NOT APPLE)
    # Linux exige estas librerías para la memoria compartida (shm_open) y hilos
    set(OS_FRONTIER_LIBS rt pthread)
else()
    # Windows maneja la memoria proyectada nativamente sin librerías extra
    set(OS_FRONTIER_LIBS "")
endif()

# 1. Definición de la ruta de destino global (Común para todos los SO)
set(GLOBAL_DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../core/utils/compiled_utils")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${GLOBAL_DESTINATION}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${GLOBAL_DESTINATION}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${GLOBAL_DESTINATION}")

# Remoción de prefijos para carga limpia en ctypes (ej: buffer_handler.dll / buffer_handler.so)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_PREFIX "")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Orden secuencial por dependencias
add_subdirectory(containers)
#add_subdirectory(locals)
add_subdirectory(remote)
add_subdirectory(buffer_handler)