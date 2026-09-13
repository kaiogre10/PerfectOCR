add_library(file_handler STATIC
	src/file_handler.cpp
)

set_target_properties(file_handler PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(file_handler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENCV_INCLUDE_DIR}
)

target_link_directories(file_handler PRIVATE
    "${LIBS_DIR}"
)

target_link_libraries(file_handler PRIVATE
    opencv_imgcodecs
    ${OS_FRONTIER_LIBS}
)
