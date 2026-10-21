# third_party/opencv_dep/deploy.cmake
cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED INSTALL_DIR OR NOT DEFINED DEPLOY_DIR)
    message(FATAL_ERROR "deploy.cmake requiere -DINSTALL_DIR= y -DDEPLOY_DIR=")
endif()

file(MAKE_DIRECTORY "${DEPLOY_DIR}")
set(_deployed 0)

if(WIN32)
    foreach(_lib core imgproc imgcodecs)
        # DLLs runtime: opencv_core4110.dll -> opencv_core.dll
        file(GLOB _dlls "${INSTALL_DIR}/bin/opencv_${_lib}*.dll")
        foreach(_f ${_dlls})
            get_filename_component(_name "${_f}" NAME)
            string(REGEX REPLACE "^opencv_${_lib}[0-9]*\\.dll$"
                                 "opencv_${_lib}.dll" _norm "${_name}")
            if(NOT _name STREQUAL _norm)
                file(COPY "${_f}" DESTINATION "${DEPLOY_DIR}")
                file(RENAME "${DEPLOY_DIR}/${_name}" "${DEPLOY_DIR}/${_norm}")
                message(STATUS " → ${_norm} (runtime, desde ${_name})")
            else()
                file(COPY "${_f}" DESTINATION "${DEPLOY_DIR}")
                message(STATUS " → ${_name} (runtime)")
            endif()
            math(EXPR _deployed "${_deployed} + 1")
        endforeach()

        # Import libs: opencv_core4110.lib -> opencv_core.lib
        file(GLOB _imps "${INSTALL_DIR}/lib/opencv_${_lib}*.lib")
        foreach(_f ${_imps})
            get_filename_component(_name "${_f}" NAME)
            string(REGEX REPLACE "^opencv_${_lib}[0-9]*\\.lib$"
                                 "opencv_${_lib}.lib" _norm "${_name}")
            if(NOT _name STREQUAL _norm)
                file(COPY "${_f}" DESTINATION "${DEPLOY_DIR}")
                file(RENAME "${DEPLOY_DIR}/${_name}" "${DEPLOY_DIR}/${_norm}")
                message(STATUS " → ${_norm} (import lib, desde ${_name})")
            else()
                file(COPY "${_f}" DESTINATION "${DEPLOY_DIR}")
                message(STATUS " → ${_name} (import lib)")
            endif()
            math(EXPR _deployed "${_deployed} + 1")
        endforeach()
    endforeach()
else()
    foreach(_lib core imgproc imgcodecs)
        set(_real "${INSTALL_DIR}/lib/libopencv_${_lib}.so.4.11.0")
        if(EXISTS "${_real}")
            file(COPY "${_real}" DESTINATION "${DEPLOY_DIR}")
            file(RENAME "${DEPLOY_DIR}/libopencv_${_lib}.so.4.11.0"
                        "${DEPLOY_DIR}/libopencv_${_lib}.so.411")
            message(STATUS " → libopencv_${_lib}.so.411")
            math(EXPR _deployed "${_deployed} + 1")
        else()
            message(WARNING "No se encontró ${_real}")
        endif()

        set(_link   "${DEPLOY_DIR}/libopencv_${_lib}.so")
        set(_target "libopencv_${_lib}.so.411")
        file(REMOVE "${_link}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink "${_target}" "${_link}"
            RESULT_VARIABLE _rc
        )
        if(_rc EQUAL 0)
            message(STATUS " → symlink: libopencv_${_lib}.so → ${_target}")
        endif()
    endforeach()
endif()

if(_deployed EQUAL 0)
    message(FATAL_ERROR "No se desplegó ningún binario a ${DEPLOY_DIR}")
endif()