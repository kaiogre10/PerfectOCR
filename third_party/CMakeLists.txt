# PerfectOCR/third_party/CMakeLists.txt
cmake_minimum_required(VERSION 3.23)
project(third_party CXX)

set(OCV_DEP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv_dep")
set(OCV_SRC     "${OCV_DEP_DIR}/src")
set(OCV_BUILD   "${OCV_DEP_DIR}/build")
set(OCV_INSTALL "${OCV_DEP_DIR}/install")
set(OCV_CONFIG  "${OCV_DEP_DIR}/opencv_config.cmake")

get_filename_component(LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libs" ABSOLUTE)

if(NOT EXISTS "${LIBS_DIR}")
    file(MAKE_DIRECTORY "${LIBS_DIR}")
endif()

# --- 0. Validaciones ---
if(NOT EXISTS "${OCV_SRC}/CMakeLists.txt")
    message(FATAL_ERROR
        "No existe ${OCV_SRC}/CMakeLists.txt.\n"
        "Ejecute primero:\n"
        "  cd ${OCV_DEP_DIR}\n"
        "  git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv.git src")
endif()

if(NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "No hay entorno Conda activo. Actívelo antes de continuar.")
endif()

# --- 1. ¿Ya está compilado e instalado? Entonces no hagas nada ---
set(_need_build TRUE)

foreach(_lib core imgproc imgcodecs)
    if(NOT EXISTS "${OCV_INSTALL}/lib/libopencv_${_lib}.so.4.11.0" AND
       NOT EXISTS "${OCV_INSTALL}/lib/libopencv_${_lib}.so.411")
        set(_need_build TRUE)
    endif()
endforeach()

if(_need_build)
    message(STATUS "PerfectOCR: OpenCV no está instalado, compilando...")

    # Solo limpia si realmente hay que recompilar
    file(REMOVE_RECURSE "${OCV_BUILD}" "${OCV_INSTALL}")

    set(ENV{OPENCV_DEP_DIR} "${OCV_DEP_DIR}")

    # Configure
    message(STATUS "PerfectOCR: configurando OpenCV")
    execute_process(
        COMMAND ${CMAKE_COMMAND}
                -C "${OCV_CONFIG}"
                -S "${OCV_SRC}"
                -B "${OCV_BUILD}"
                -G Ninja
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "Fallo el configure de OpenCV (código ${_rc})")
    endif()

    # Build (dinámico, Ninja decide)
    message(STATUS "PerfectOCR: compilando OpenCV")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build "${OCV_BUILD}" --config Release --parallel
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "Fallo la compilación de OpenCV (código ${_rc})")
    endif()

    # Install
    message(STATUS "PerfectOCR: instalando OpenCV en ${OCV_INSTALL}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --install "${OCV_BUILD}"
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "Fallo la instalación de OpenCV (código ${_rc})")
    endif()
else()
    message(STATUS "PerfectOCR: OpenCV ya está instalado en ${OCV_INSTALL}, saltando compilación")
endif()

# --- 2. Deploy a PerfectOCR/libs (siempre, por si acaso) ---
message(STATUS "PerfectOCR: desplegando binarios en ${LIBS_DIR}")
file(MAKE_DIRECTORY "${LIBS_DIR}")

set(_deployed 0)

if(WIN32)
    foreach(_lib core imgproc imgcodecs)
        file(GLOB _dlls "${OCV_INSTALL}/bin/opencv_${_lib}*.dll")
        foreach(_f ${_dlls})
            file(COPY "${_f}" DESTINATION "${LIBS_DIR}")
            get_filename_component(_name "${_f}" NAME)
            message(STATUS "  → ${_name} (runtime)")
            math(EXPR _deployed "${_deployed} + 1")
        endforeach()
        file(GLOB _imps "${OCV_INSTALL}/lib/opencv_${_lib}[0-9]*.lib")
        foreach(_f ${_imps})
            file(COPY "${_f}" DESTINATION "${LIBS_DIR}")
            get_filename_component(_name "${_f}" NAME)
            message(STATUS "  → ${_name} (import lib)")
            math(EXPR _deployed "${_deployed} + 1")
        endforeach()
    endforeach()
else()
    foreach(_lib core imgproc imgcodecs)
        set(_real "${OCV_INSTALL}/lib/libopencv_${_lib}.so.4.11.0")
        if(EXISTS "${_real}")
            file(COPY "${_real}" DESTINATION "${LIBS_DIR}")
            file(RENAME "${LIBS_DIR}/libopencv_${_lib}.so.4.11.0"
                        "${LIBS_DIR}/libopencv_${_lib}.so.411")
            message(STATUS "  → libopencv_${_lib}.so.411")
            math(EXPR _deployed "${_deployed} + 1")
        else()
            message(WARNING "No se encontró ${_real}")
        endif()

        set(_link   "${LIBS_DIR}/libopencv_${_lib}.so")
        set(_target "libopencv_${_lib}.so.411")
        file(REMOVE "${_link}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink "${_target}" "${_link}"
            RESULT_VARIABLE _rc
        )
        if(_rc EQUAL 0)
            message(STATUS "  → symlink: libopencv_${_lib}.so → ${_target}")
        endif()
    endforeach()
endif()

if(_deployed EQUAL 0)
    message(FATAL_ERROR "No se desplegó ningún binario a ${LIBS_DIR}.")
endif()

message(STATUS "PerfectOCR: listo. Headers: ${OCV_INSTALL}/include/opencv4")