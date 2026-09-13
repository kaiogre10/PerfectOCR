add_library(buffer_handler SHARED
    src/buffer_handler.cpp
    include/buffer_handler.h
)

set_target_properties(buffer_handler PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(buffer_handler PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(buffer_handler PRIVATE 
    payload_container
    image_container
    ${OS_FRONTIER_LIBS}
)
