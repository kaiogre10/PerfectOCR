# Compilador Intel oneAPI
set(CMAKE_C_COMPILER "icx" CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER "icx" CACHE STRING "" FORCE)

# Forzar flags de arquitectura AVX2 para que Clang habilite el target feature
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2" CACHE STRING "" FORCE)

# Compatibilidad con MSVC 1951 / VS 2026
set(OpenCV_RUNTIME "vc18" CACHE STRING "" FORCE)
set(OpenCV_ARCH "x64" CACHE STRING "" FORCE)

# Configuración de instalación
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/install" CACHE PATH "" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)

# Sistema moderno de optimización de CPU de OpenCV
set(CPU_BASELINE "AVX2" CACHE STRING "Fijar AVX2 en todo el codigo base" FORCE)
set(CPU_DISPATCH "" CACHE STRING "Desactivar generacion dinamica de variantes" FORCE)

# Desactivar PCH para evitar conflictos de Clang en Windows
set(ENABLE_PRECOMPILED_HEADERS OFF CACHE BOOL "" FORCE)
set(OPENCV_WARNINGS_ARE_ERRORS OFF CACHE BOOL "" FORCE)

# Módulos requeridos
set(BUILD_LIST "core,imgproc,imgcodecs" CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# Librerías de aceleración Intel
set(WITH_TBB ON CACHE BOOL "" FORCE)
set(WITH_OPENMP OFF CACHE BOOL "" FORCE)
set(WITH_IPP ON CACHE BOOL "" FORCE)

# Códecs
set(BUILD_PNG ON CACHE BOOL "" FORCE)
set(BUILD_JPEG ON CACHE BOOL "" FORCE)
set(BUILD_TIFF ON CACHE BOOL "" FORCE)
set(BUILD_WEBP ON CACHE BOOL "" FORCE)

# Desactivar componentes no requeridos
set(WITH_CUDA OFF CACHE BOOL "" FORCE)
set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
set(WITH_GSTREAMER OFF CACHE BOOL "" FORCE)
set(WITH_MSMF OFF CACHE BOOL "" FORCE)
set(WITH_DSHOW OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_java OFF CACHE BOOL "" FORCE)