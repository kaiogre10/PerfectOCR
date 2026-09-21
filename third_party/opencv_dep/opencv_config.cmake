# third_party/opencv_dep/opencv_config.cmake
if(WIN32 OR CMAKE_HOST_WIN32)
    if(NOT CMAKE_GENERATOR MATCHES "Visual Studio")
        set(CMAKE_C_COMPILER "icx" CACHE STRING "" FORCE)
        set(CMAKE_CXX_COMPILER "icx" CACHE STRING "" FORCE)
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2" CACHE STRING "" FORCE)

    set(OpenCV_RUNTIME "vc18" CACHE STRING "" FORCE)
    set(OpenCV_ARCH "x64"  CACHE STRING "" FORCE)

else()
    # 1. Descubrir CONDA_ENV del entorno (no hardcodear)
    if(NOT DEFINED CONDA_ENV)

		message("AMBIENTE DE CONDA ENCONTRADO: ${CONDA_PREFIX}")
        if(DEFINED CONDA_PREFIX)
            set(CONDA_ENV CONDA_PREFIX)
        else()
            message(FATAL_ERROR
                "${CONDA_ENV} no definido y ${CONDA_PREFIX} no está en el entorno "
                "Activa el env con `conda activate intel`.")
        endif()
    endif()

    # 2. Descubrir GCC install dir dinámicamente
    file(GLOB GCC_VERSIONS "$ENV{CONDA_PREFIX}/bin/gcc")
    if(GCC_VERSIONS)
        list(SORT GCC_VERSIONS ORDER DESCENDING)
        list(GET GCC_VERSIONS 0 GCC_INSTALL_PATH)
        message(STATUS "GCC install dir: ${GCC_INSTALL_PATH} ")
    else()
        message(FATAL_ERROR
            "No se encontró GCC en: $ENV{CONDA_PREFIX}/bin/gcc"
            "Verifica que el env de conda tenga gcc instalado.")
    endif()

    # 3. Flags de compilación
    set(CMAKE_C_FLAGS "-mavx2 -mfma --gcc-install-dir=${GCC_INSTALL_PATH} -L${CONDA_ENV}/lib" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS "-mavx2 -mfma --gcc-install-dir=${GCC_INSTALL_PATH} -L${CONDA_ENV}/lib" CACHE STRING "" FORCE)

    # 4. Flags de enlace
    set(CMAKE_EXE_LINKER_FLAGS "-L${CONDA_ENV}/lib -Wl,-rpath,${CONDA_ENV}/lib" CACHE STRING "" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS "-L${CONDA_ENV}/lib -Wl,-rpath,${CONDA_ENV}/lib" CACHE STRING "" FORCE)
endif()

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)

# Vectorización estricta AVX2 sin generación dinámica de variantes
set(CPU_BASELINE "AVX2" CACHE STRING "Fijar AVX2 en todo el codigo base" FORCE)
set(CPU_DISPATCH "" CACHE STRING "Desactivar generacion dinamica de variantes" FORCE)

# Control de PCH y advertencias
set(ENABLE_PRECOMPILED_HEADERS OFF CACHE BOOL "" FORCE)
set(OPENCV_WARNINGS_ARE_ERRORS OFF CACHE BOOL "" FORCE)

# Módulos requeridos
set(BUILD_LIST "core,imgproc,imgcodecs" CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# Optimizaciones de hardware Intel
set(WITH_IPP ON CACHE BOOL "" FORCE)
set(WITH_TBB ON CACHE BOOL "" FORCE)
set(WITH_OPENMP OFF CACHE BOOL "" FORCE)

# Códecs
set(BUILD_PNG ON CACHE BOOL "" FORCE)
set(BUILD_JPEG ON CACHE BOOL "" FORCE)
set(BUILD_TIFF ON CACHE BOOL "" FORCE)
set(BUILD_WEBP ON CACHE BOOL "" FORCE)

# Desactivar componentes innecesarios
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