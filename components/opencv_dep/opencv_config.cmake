if(WIN32 OR CMAKE_HOST_WIN32)
    # Compiladores Intel oneAPI en Windows
    set(CMAKE_C_COMPILER "icx" CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER "icx" CACHE STRING "" FORCE)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2" CACHE STRING "" FORCE)

    set(OpenCV_RUNTIME "vc18" CACHE STRING "" FORCE)
    set(OpenCV_ARCH "x64" CACHE STRING "" FORCE)
else()
    # Compiladores Intel oneAPI en Linux
    set(CMAKE_C_COMPILER "icx" CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER "icpx" CACHE STRING "" FORCE)

    set(CONDA_ENV "/home/kaiogre05/miniforge3/envs/intel")
    set(GCC_INSTALL_PATH "${CONDA_ENV}/lib/gcc/x86_64-conda-linux-gnu/15.2.0")

    # Banderas de compilación y enlace
    set(CMAKE_C_FLAGS "-mavx2 -mfma --gcc-install-dir=${GCC_INSTALL_PATH} -L${CONDA_ENV}/lib" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS "-mavx2 -mfma --gcc-install-dir=${GCC_INSTALL_PATH} -L${CONDA_ENV}/lib" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS "-L${CONDA_ENV}/lib -Wl,-rpath,${CONDA_ENV}/lib" CACHE STRING "" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS "-L${CONDA_ENV}/lib -Wl,-rpath,${CONDA_ENV}/lib" CACHE STRING "" FORCE)
endif()

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/install" CACHE PATH "" FORCE)
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
set(WITH_TBB ON CACHE BOOL "" FORCE)
set(WITH_OPENMP OFF CACHE BOOL "" FORCE)
set(WITH_IPP ON CACHE BOOL "" FORCE)

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