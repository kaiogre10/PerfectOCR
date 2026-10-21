# PerfectOCR/third_party/opencv_dep/CMakeLists.txt
cmake_minimum_required(VERSION 3.23)
project(opencv_dep CXX)

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
set(OCV_CONFIG "${DEP_SRC_DIR}/opencv_config.cmake")
set(OCV_LIBS_DIR "${DEP_DEPLOY_DIR}")

file(MAKE_DIRECTORY "${OCV_LIBS_DIR}")

if(NOT EXISTS "${OCV_SRC}/CMakeLists.txt")
    message(FATAL_ERROR
        "Falta ${OCV_SRC}/CMakeLists.txt. Ejecuta:\n"
        "  cd ${DEP_SRC_DIR}\n"
        "  git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv.git src")
endif()

set(_ocv_sentinel "${OCV_INSTALL}/.done")

if(EXISTS "${_ocv_sentinel}")
    message(STATUS "opencv_dep: ya instalado, saltando configure/build/install")
else()
    set(ENV{OPENCV_DEP_DIR} "${DEP_SRC_DIR}")

    message(STATUS "opencv_dep: configure")
    execute_process(
        COMMAND ${CMAKE_COMMAND}
                -C "${OCV_CONFIG}"
                -S "${OCV_SRC}"
                -B "${OCV_OPENCV_B}"
                -G Ninja
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