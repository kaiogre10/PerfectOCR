add_library(payload_container STATIC
    src/payload_container.cpp
)

set_target_properties(payload_container PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(payload_container PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(payload_container PUBLIC
    ${OS_FRONTIER_LIBS}
)