find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import cython; print(cython.__version__)"
    OUTPUT_VARIABLE CYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT CYTHON_VERSION)
    message(FATAL_ERROR "cython no está instalado en ${Python3_EXECUTABLE}. Instálalo con: pip install cython")
endif()

message(STATUS "cython version: ${CYTHON_VERSION}")

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

set(COMPILED_PYTHON_EXT_SUFFIX "${PYTHON_EXT_SUFFIX}")
set(COMPILED_NUMPY_INCLUDE_DIR "${NUMPY_INCLUDE_DIR}")
set(COMPILED_PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
set(COMPILED_IMAGE_CONTAINER_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/components/image_container/include"
)

function(configure_compiled_target target)
    set_target_properties(${target} PROPERTIES
        PREFIX ""
        SUFFIX "${COMPILED_PYTHON_EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    foreach(CONFIG DEBUG RELEASE RELWITHDEBINFO)
        string(TOUPPER "${CONFIG}" CONFIG_UPPER)
        set_target_properties(${target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER}
            "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endforeach()

    if(UNIX)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/libs;${CMAKE_SOURCE_DIR}/libraries"
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/libs;${CMAKE_SOURCE_DIR}/libraries"
        )
    endif()

    target_include_directories(${target} PRIVATE
        ${COMPILED_PYTHON_INCLUDE_DIRS}
        "${COMPILED_NUMPY_INCLUDE_DIR}"
        "${COMPILED_IMAGE_CONTAINER_INCLUDE_DIR}"
    )

    target_link_directories(${target} PRIVATE
        "${CMAKE_SOURCE_DIR}/libraries"
        "${CMAKE_SOURCE_DIR}/libs"
    )

    if(WIN32)
        target_compile_options(${target} PRIVATE /std:c++20 /arch:AVX2 /O2)
    else()
        target_compile_options(${target} PRIVATE -std=c++20 -mavx2 -mf16c -mfma)
    endif()
endfunction()

add_subdirectory(image)
add_subdirectory(arrays)

add_dependencies(image arrays)
