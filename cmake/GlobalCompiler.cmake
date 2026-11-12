# cmake/GlobalCompilerOptions.cmake
cmake_minimum_required(VERSION 3.23)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Fuerza la compilación de librerías dinámicas" ON)

if (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
	message(STATUS "Intel OneAPI detectado. Activando optimización de arquitectura host.")
	if (WIN32)
		add_compile_options(
			/arch:AVX2
			/W4
		)
		set(OS_FRONTIER_LIBS "")
		# Optimizaciones para icx-cl en Windows
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O3 /QxHost")
		set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O3 /QxHost")
		# ELIMINACIÓN OBLIGATORIA: Purga de /Qm64 y flags obsoletos de Intel clásico
		foreach(flag_var
			CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
			CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
			CMAKE_EXE_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS_DEBUG CMAKE_EXE_LINKER_FLAGS_RELEASE
			CMAKE_SHARED_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS_DEBUG CMAKE_SHARED_LINKER_FLAGS_RELEASE)
			if(DEFINED ${flag_var})
				string(REPLACE "/Qm64" "" ${flag_var} "${${flag_var}}")
				string(REPLACE "-qm64" "" ${flag_var} "${${flag_var}}")
				string(REPLACE "/Qnextgen" "" ${flag_var} "${${flag_var}}")
				string(REPLACE "-qnextgen" "" ${flag_var} "${${flag_var}}")
			endif()
		endforeach()
	else ()
		# Sintaxis GNU/Clang para Linux
		set(OS_FRONTIER_LIBS rt pthread)
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -xHost")
		set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -xHost")
		add_compile_options(
			-mavx2
			-mf16c
			-mfma
			-Wall
			-Wextra
			-Wpedantic
		)
	endif ()
endif ()
