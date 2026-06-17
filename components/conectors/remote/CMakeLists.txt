add_library(remote SHARED
    "remote_conectors.h"
    "conectors_net.cpp"
)

# Exposición de interfaces del conector
target_include_directories(remote 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
