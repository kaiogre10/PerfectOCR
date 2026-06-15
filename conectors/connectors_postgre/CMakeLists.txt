# CMakeList.txt: proyecto de CMake para connectors_postgre, incluya el origen y defina
# la lógica específica del proyecto aquí.
#

# Agregue un origen al ejecutable de este proyecto.
add_executable (connectors_postgre "connectors_postgre.cpp" "connectors_postgre.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET connectors_postgre PROPERTY CXX_STANDARD 20)
endif()

# TODO: Agregue pruebas y destinos de instalación si es necesario.
