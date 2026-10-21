# PerfectOCR/third_party/xtensor_dep/CMakeLists.txt
cmake_minimum_required(VERSION 3.23)
project(xtensor_dep CXX)

if(NOT DEFINED DEP_DEPLOY_DIR)
    get_filename_component(DEP_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
    set(DEP_INSTALL_DIR "${DEP_SRC_DIR}/install")
    get_filename_component(_libs "${DEP_SRC_DIR}/../../libs" ABSOLUTE)
    set(DEP_DEPLOY_DIR "${_libs}/xtensor_dep")
endif()

set(XT_SRC "${DEP_SRC_DIR}/src")
set(XTL_SRC "${DEP_SRC_DIR}/src/xtl")
set(XSIMD_SRC "${DEP_SRC_DIR}/src/xsimd")
set(XT_INSTALL_INC "${DEP_INSTALL_DIR}/include")

if(NOT EXISTS "${XT_SRC}/include/xtensor/containers/xarray.hpp")
    message(FATAL_ERROR
        "Falta ${XT_SRC}. Ejecuta:\n"
        "  cd ${DEP_SRC_DIR}\n"
        "  git clone --depth 1 --branch 0.27.1 https://github.com/xtensor-stack/xtensor.git src\n"
        "  git clone --depth 1 --branch 0.8.2  https://github.com/xtensor-stack/xtl.git src/xtl\n"
        "  git clone --depth 1 --branch 13.0.0 https://github.com/xtensor-stack/xsimd.git src/xsimd")
endif()

set(_sentinel "${DEP_INSTALL_DIR}/.done")

if(NOT EXISTS "${_sentinel}")
    message(STATUS "xtensor_dep: instalando headers en ${XT_INSTALL_INC}")
    file(MAKE_DIRECTORY "${XT_INSTALL_INC}")

    file(COPY "${XT_SRC}/include/" DESTINATION "${XT_INSTALL_INC}")
    file(COPY "${XTL_SRC}/include/" DESTINATION "${XT_INSTALL_INC}")
    file(COPY "${XSIMD_SRC}/include/" DESTINATION "${XT_INSTALL_INC}")

    file(WRITE "${_sentinel}" "ok\n")
else()
    message(STATUS "xtensor_dep: ya instalado (${_sentinel}), saltando")
endif()

add_custom_target(deploy
    COMMAND ${CMAKE_COMMAND} -E echo
            "xtensor_dep: header-only, sin binarios que deployar a libs/"
    COMMENT "xtensor_dep: deploy no-op (headers en ${XT_INSTALL_INC})"
    VERBATIM
)
