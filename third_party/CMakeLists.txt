# PerfectOCR/third_party/CMakeLists.txt
cmake_minimum_required(VERSION 3.23)
project(third_party CXX)

if(NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "No hay entorno Conda activo. Actívelo antes de continuar.")
endif()

if(WIN32 OR CMAKE_HOST_WIN32)
    if(NOT CMAKE_GENERATOR MATCHES "Visual Studio")
        set(CMAKE_C_COMPILER "icx" CACHE STRING "" FORCE)
        set(CMAKE_CXX_COMPILER "icx" CACHE STRING "" FORCE)
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2" CACHE STRING "" FORCE)
else()
    set(CMAKE_C_COMPILER "icx" CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER "icpx" CACHE STRING "" FORCE)
    set(CMAKE_C_FLAGS "-mavx2 -mf16c -mfma" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS "-mavx2 -mf16c -mfma" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS "-L${CONDA_ENV}/lib -Wl,-rpath,${CONDA_ENV}/lib" CACHE STRING "" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS "-L${CONDA_ENV}/lib -Wl,-rpath,${CONDA_ENV}/lib" CACHE STRING "" FORCE)
endif()

get_filename_component(LIBS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libs" ABSOLUTE)
file(MAKE_DIRECTORY "${LIBS_ROOT}")

# Autodetección o lista explícita vía -DTHIRD_PARTY_DEPS="opencv_dep;xtensor_dep"
set(THIRD_PARTY_DEPS "" CACHE STRING
    "Dependencias a compilar (vacío = autodetectar subcarpetas con CMakeLists.txt)")

if(THIRD_PARTY_DEPS STREQUAL "")
    file(GLOB _entries LIST_DIRECTORIES true
         RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
         "${CMAKE_CURRENT_SOURCE_DIR}/*")
    foreach(_e ${_entries})
        if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_e}"
           AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_e}/CMakeLists.txt"
           AND NOT _e STREQUAL "build"
           AND NOT _e STREQUAL "install")
            list(APPEND THIRD_PARTY_DEPS "${_e}")
        endif()
    endforeach()
endif()

if(NOT THIRD_PARTY_DEPS)
    message(WARNING "No se detectaron dependencias en ${CMAKE_CURRENT_SOURCE_DIR}")
    return()
endif()

message(STATUS "PerfectOCR: dependencias: ${THIRD_PARTY_DEPS}")

foreach(_dep ${THIRD_PARTY_DEPS})
    set(DEP_NAME "${_dep}")
    set(DEP_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${_dep}")
    set(DEP_BUILD_DIR "${DEP_SRC_DIR}/build")
    set(DEP_INSTALL_DIR "${DEP_SRC_DIR}/install")
    set(DEP_DEPLOY_DIR "${LIBS_ROOT}/${_dep}")

    if(NOT EXISTS "${DEP_SRC_DIR}/CMakeLists.txt")
        message(WARNING "Saltando ${_dep}: sin CMakeLists.txt")
        continue()
    endif()

    message(STATUS "PerfectOCR: ── ${_dep} ──")

    set(_sentinel "${DEP_INSTALL_DIR}/.done")

    if(EXISTS "${_sentinel}")
        message(STATUS "${_dep}: ya instalado (${_sentinel}), saltando build")
    else()
        message(STATUS "${_dep}: configure")
        execute_process(
            COMMAND ${CMAKE_COMMAND}
                    -S "${DEP_SRC_DIR}"
                    -B "${DEP_BUILD_DIR}"
                    -G Ninja
                    -DDEP_NAME=${DEP_NAME}
                    -DDEP_SRC_DIR=${DEP_SRC_DIR}
                    -DDEP_BUILD_DIR=${DEP_BUILD_DIR}
                    -DDEP_INSTALL_DIR=${DEP_INSTALL_DIR}
                    -DDEP_DEPLOY_DIR=${DEP_DEPLOY_DIR}
                    -DLIBS_ROOT=${LIBS_ROOT}
            COMMAND_ECHO STDOUT
            RESULT_VARIABLE _rc
        )
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "${_dep}: fallo configure (código ${_rc})")
        endif()

        message(STATUS "${_dep}: build")
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build "${DEP_BUILD_DIR}" --config Release --parallel
            COMMAND_ECHO STDOUT
            RESULT_VARIABLE _rc
        )
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "${_dep}: fallo build (código ${_rc})")
        endif()

        message(STATUS "${_dep}: install")
        execute_process(
            COMMAND ${CMAKE_COMMAND} --install "${DEP_BUILD_DIR}" --config Release
            COMMAND_ECHO STDOUT
            RESULT_VARIABLE _rc
        )
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "${_dep}: fallo install (código ${_rc})")
        endif()

        file(WRITE "${_sentinel}" "ok\n")
    endif()

    # Redeploy siempre (barato: copia de archivos ya compilados)
    message(STATUS "${_dep}: deploy -> ${DEP_DEPLOY_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND}
                --build "${DEP_BUILD_DIR}"
                --config Release
                --target deploy
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "${_dep}: fallo deploy (código ${_rc})")
    endif()

    message(STATUS "PerfectOCR: ── ${_dep} OK ──")
endforeach()

message(STATUS "PerfectOCR: listo. libs/: ${LIBS_ROOT}")