add_library(locals SHARED
    "conectors_postgre.cpp"
    "local_conectors.h"
)

# Exposición de interfaces del conector
target_include_directories(locals 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
