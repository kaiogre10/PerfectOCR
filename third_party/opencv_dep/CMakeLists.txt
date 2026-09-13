cmake_minimum_required(VERSION 3.20)
project(opencv_dep CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/opencv_config.cmake")

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)

add_subdirectory(src)

set(LIBS_DEST "${CMAKE_CURRENT_SOURCE_DIR}/../../libs")

add_custom_target(copy_opencv_to_libs ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBS_DEST}"
    COMMENT "Copiando OpenCV a ${LIBS_DEST}"

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:opencv_core>" "${LIBS_DEST}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:opencv_imgproc>" "${LIBS_DEST}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:opencv_imgcodecs>" "${LIBS_DEST}/"

    DEPENDS opencv_core opencv_imgproc opencv_imgcodecs
)

if(WIN32)
    add_custom_command(TARGET copy_opencv_to_libs POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_LINKER_FILE:opencv_core>" "${LIBS_DEST}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_LINKER_FILE:opencv_imgproc>" "${LIBS_DEST}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_LINKER_FILE:opencv_imgcodecs>" "${LIBS_DEST}/"
    )
endif()