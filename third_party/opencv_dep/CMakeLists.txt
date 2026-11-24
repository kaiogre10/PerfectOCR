# PerfectOCR/third_party/opencv_dep/CMakeLists.txt
cmake_minimum_required(VERSION 3.25)
project(opencv_dep LANGUAGES C CXX)

set(OCV_CONFIG "${CMAKE_CURRENT_LIST_DIR}/opencv_config.cmake" CACHE FILEPATH "" FORCE)

set(ENV{OPENCV_DEP_DIR} "${CMAKE_CURRENT_LIST_DIR}")

include("${OCV_CONFIG}")

message(STATUS "OPENCV DEP INICIANDO")
message(STATUS "Compilador C: ${CMAKE_C_COMPILER}")
message(STATUS "Compilador C++: ${CMAKE_CXX_COMPILER}")

if(NOT DEFINED DEP_DEPLOY_DIR)
    get_filename_component(DEP_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
    set(DEP_BUILD_DIR "${DEP_SRC_DIR}/build")
    set(DEP_INSTALL_DIR "${DEP_SRC_DIR}/install")
    get_filename_component(_libs "${DEP_SRC_DIR}/../../libs" ABSOLUTE)
    set(DEP_DEPLOY_DIR "${_libs}/opencv_dep")
endif()

set(OCV_SRC "${DEP_SRC_DIR}/src")
set(OCV_OPENCV_B "${DEP_BUILD_DIR}/opencv")
set(OCV_INSTALL "${DEP_INSTALL_DIR}")
set(OCV_LIBS_DIR "${DEP_DEPLOY_DIR}")

file(MAKE_DIRECTORY "${OCV_LIBS_DIR}")

if(NOT EXISTS "${OCV_SRC}/CMakeLists.txt")
    message(FATAL_ERROR
        "Falta ${OCV_SRC}/CMakeLists.txt. Ejecuta:\n"
        "  cd ${DEP_SRC_DIR}\n"
        "  git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv.git src")
endif()

# FIX 1: recuperar _PROJECT_RULES_CMAKE si viene vacío
if(NOT DEFINED _PROJECT_RULES_CMAKE OR _PROJECT_RULES_CMAKE STREQUAL "")
    get_filename_component(_PROJECT_RULES_CMAKE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CompilerFlags.cmake" ABSOLUTE)
endif()
message(STATUS "opencv_dep: PROJECT_RULES = ${_PROJECT_RULES_CMAKE}")
if(NOT EXISTS "${_PROJECT_RULES_CMAKE}")
    message(FATAL_ERROR "No existe ${_PROJECT_RULES_CMAKE}")
endif()

set(_ocv_sentinel "${OCV_INSTALL}/.done")

if(EXISTS "${_ocv_sentinel}")
    message(STATUS "opencv_dep: ya instalado, saltando configure/build/install")
else()
    set(ENV{OPENCV_DEP_DIR} "${DEP_SRC_DIR}")

    # FIX 2: flags que OpenCV SÍ respeta (concatena, no pisa)
    if(CMAKE_CXX_COMPILER MATCHES "icpx$" OR CMAKE_C_COMPILER MATCHES "icx$")
        set(_OCV_ISA_FLAGS "-mavx2 -mf16c -mfma")
        message(STATUS "opencv_dep: OneAPI detectado, ISA flags = ${_OCV_ISA_FLAGS}")
    else()
        set(_OCV_ISA_FLAGS "-mavx2 -mf16c -mfma")
        message(STATUS "opencv_dep: compilador no-OneAPI, ISA flags = ${_OCV_ISA_FLAGS}")
    endif()
    set(_OCV_RELEASE_FLAGS "-O3 ${_OCV_ISA_FLAGS}")

    message(STATUS "OPEC_CV DEP ESTADO: configure")
    execute_process(
        COMMAND ${CMAKE_COMMAND}
                -G "${CMAKE_GENERATOR}"
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
                -DCMAKE_PREFIX_PATH=$ENV{CONDA_PREFIX}
                -DCMAKE_PROJECT_INCLUDE=${_PROJECT_RULES_CMAKE}
                # FIX 3: pasar flags por la vía que OpenCV respeta
                -DOPENCV_EXTRA_FLAGS_RELEASE=${_OCV_RELEASE_FLAGS}
                -DOPENCV_EXTRA_C_FLAGS_RELEASE=${_OCV_RELEASE_FLAGS}
                -DOPENCV_EXTRA_CXX_FLAGS_RELEASE=${_OCV_RELEASE_FLAGS}
                # FIX 4: CPU_BASELINE con tokens de OpenCV (no flags GCC)
                "-DCPU_BASELINE=AVX2;FMA3;FP16"
                -DCPU_DISPATCH=
                -C "${OCV_CONFIG}"
                -S "${OCV_SRC}"
                -B "${OCV_OPENCV_B}"
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "opencv_dep: fallo configure (${_rc})")
    endif()

    message(STATUS "opencv_dep: build")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build "${OCV_OPENCV_B}" --config Release --parallel
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "opencv_dep: fallo build (${_rc})")
    endif()

    message(STATUS "opencv_dep: install")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --install "${OCV_OPENCV_B}" --config Release
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "opencv_dep: fallo install (${_rc})")
    endif()

    file(WRITE "${_ocv_sentinel}" "ok\n")
endif()

add_custom_target(deploy
    COMMAND ${CMAKE_COMMAND}
            -DINSTALL_DIR=${OCV_INSTALL}
            -DDEPLOY_DIR=${OCV_LIBS_DIR}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/deploy.cmake"
    COMMENT "opencv_dep: deploy -> ${OCV_LIBS_DIR}"
    VERBATIM
)