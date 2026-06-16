add_library(connectors_postgre SHARED
    "connectors_postgre.cpp"
    "connectors_postgre.h"
)

# Exposición de interfaces del conector
target_include_directories(connectors_postgre 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Descomentar si el conector requiere consumir tipos de containers o buffer_handler:
# target_link_libraries(connectors_postgre 
#     PRIVATE 
#         containers
#         buffer_handler
# )