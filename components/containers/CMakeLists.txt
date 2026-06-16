add_library(containers SHARED 
    "containers.h" 
    "payload_container.cpp"
)

target_include_directories(containers 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)