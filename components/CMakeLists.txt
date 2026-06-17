cmake_minimum_required(VERSION 3.20)
project(components LANGUAGES CXX)

# Políticas y estándar
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BUILD_SHARED_LIBS "Fuerza la compilación de librerías dinámicas" ON)

# Configuración de salida para los binarios de ctypes
# 1. Definición de la ruta de destino global (Común para todos los SO)
set(GLOBAL_DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../core/utils/compiled_utils")

# 2. Configuración general de directorios de salida
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${GLOBAL_DESTINATION}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${GLOBAL_DESTINATION}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${GLOBAL_DESTINATION}")

# ============================================================================
# CONFIGURACIÓN CONDICIONAL POR SISTEMA OPERATIVO
# ============================================================================

if(WIN32)
    # --- Bloque Exclusivo para Windows ---
    message(STATUS "Configurando entorno de compilacion nativo para Windows...")

    # Quita el prefijo 'lib' a las DLLs y archivos de importación en Windows
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
    set(CMAKE_IMPORT_LIBRARY_PREFIX "")

    # Exporta automáticamente todos los símbolos (Evita usar __declspec(dllexport) a mano)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

elseif(UNIX AND NOT APPLE)
    # --- Bloque Exclusivo para Linux ---
    message(STATUS "Configurando entorno de compilacion para Linux...")
    
    # Aquí puedes agregar banderas específicas de Linux si las necesitas en el futuro

endif()

# Orden secuencial por dependencias
add_subdirectory(containers)
add_subdirectory(buffer_handler)
#add_subdirectory(conectors)