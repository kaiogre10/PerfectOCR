add_library(remote SHARED
    "remote_conectors.h"
    "conectors_net.cpp"
)

target_include_directories(remote
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)