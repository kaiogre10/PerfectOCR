cmake_minimum_required(VERSION 3.20)
project(opencv_dep CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/opencv_config.cmake")
add_subdirectory(src)

#set(LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../libs")

add_custom_target(copy_opencv_to_libs ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${LIBS_DIR}"
    COMMENT "Copiando 3 .so de OpenCV a ${LIBS_DIR}"

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_SONAME_FILE:opencv_core>" "${LIBS_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_SONAME_FILE:opencv_imgproc>" "${LIBS_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_SONAME_FILE:opencv_imgcodecs>" "${LIBS_DIR}/"

    DEPENDS opencv_core opencv_imgproc opencv_imgcodecs
)