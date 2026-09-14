find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# ─── Verificar Cython (via Python) ───
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import Cython; print(Cython.__version__)"
    OUTPUT_VARIABLE CYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT CYTHON_VERSION)
    message(FATAL_ERROR "Cython no está instalado en ${Python3_EXECUTABLE}. Instálalo con: pip install cython")
endif()

message(STATUS "Cython version: ${CYTHON_VERSION}")

# ─── Obtener EXT_SUFFIX y NumPy include ───
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# ─── Cython: image.pyx → image.cpp ───
add_custom_command(
    OUTPUT  "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp"
    COMMAND ${Python3_EXECUTABLE} -m cython
            --cplus -3
            "${CMAKE_CURRENT_SOURCE_DIR}/image.pyx"
            -o "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/image.pyx"
            "${CMAKE_CURRENT_SOURCE_DIR}/image.pxd"
            "${CMAKE_CURRENT_SOURCE_DIR}/image_loader.pxd"
    COMMENT "Cython: image.pyx → image.cpp"
    VERBATIM
)

# ─── Módulo Python ───
add_library(image MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp"
)

set_target_properties(image PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
foreach(CONFIG DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set_target_properties(image PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endforeach()
# ─── RPATH (Linux) ───
if(UNIX)
    set_target_properties(image PROPERTIES
        BUILD_RPATH   "${CMAKE_SOURCE_DIR}/../libs;${CMAKE_SOURCE_DIR}/../libraries"
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/../libs;${CMAKE_SOURCE_DIR}/../libraries"
    )
endif()

# ─── Includes ───
target_include_directories(image PRIVATE
    ${Python3_INCLUDE_DIRS}
    "${NUMPY_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/image_loader/include"
    "${CMAKE_SOURCE_DIR}/image_container/include"
    "${CMAKE_SOURCE_DIR}/file_handler/include"
    "${CMAKE_SOURCE_DIR}/c_utils/include"
    "${CMAKE_SOURCE_DIR}/../third_party/opencv_dep/install/include/opencv4"
)

# ─── Link ───
target_link_directories(image PRIVATE
    "${CMAKE_SOURCE_DIR}/../libraries"
    "${CMAKE_SOURCE_DIR}/../libs"
)

target_link_libraries(image PRIVATE
    image_loader
    image_container
    file_handler
    c_utils
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)

# ─── Flags por OS ───
if(WIN32)
    target_compile_options(image PRIVATE /std:c++20 /arch:AVX2 /O2)
else()
    target_compile_options(image PRIVATE -std=c++20 -mavx2 -mfma)
endif()