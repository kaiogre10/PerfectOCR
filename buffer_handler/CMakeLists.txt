cmake_minimum_required(VERSION 3.20)
project("buffer_handler" LANGUAGES CXX)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<xIF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(buffer_handler SHARED 
    "buffer_handler.cpp"
    "buffer_handler.h"
)

set(DESTINY "${CMAKE_CURRENT_SOURCE_DIR}/../core/utils/compiled_utils")

# Las rutas de salida hacia Cython se mantienen intactas
set_target_properties(buffer_handler PROPERTIES
        PREFIX ""
        WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if (WIN32)
  set_target_properties(buffer_handler PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${DESTINY}"
  ARCHIVE_OUTPUT_DIRECTORY "${DESTINY}"
  )

else()
  set_target_properties(buffer_handler PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DESTINY}"
  )

endif()
