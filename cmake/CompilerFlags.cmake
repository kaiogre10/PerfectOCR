# cmake/CompilerFlags.cmake
cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Fuerza la compilación de librerías dinámicas" ON)

set(OS_FRONTIER_LIBS "")

if(UNIX AND NOT WIN32)
    set(OS_FRONTIER_LIBS rt pthread)
    
    set(CMAKE_EXE_LINKER_FLAGS "--ld-path=${CMAKE_LINKER} -Wl,--as-needed -Wl,--gc-sections -Wl,--icf=safe")
    set(CMAKE_SHARED_LINKER_FLAGS "--ld-path=${CMAKE_LINKER} -Wl,--no-undefined -Wl,--as-needed -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,--icf=safe")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS " --ld-path=${CMAKE_LINKER} -Wl,--gc-sections -Wl,--icf=safe")

    set(CMAKE_C_FLAGS_RELEASE
        "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG -mavx2 -mf16c -mfma"
    )
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -mavx2 -mf16c -mfma"
    )

    set(CMAKE_STATIC_LINKER_FLAGS "")

elseif(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    set(CMAKE_C_FLAGS_RELEASE
            "${CMAKE_C_FLAGS_RELEASE} /O3 /DNDEBUG /MD /arch:AVX2 /Gy /Gw /fp:precise"
    )
    set(CMAKE_CXX_FLAGS_RELEASE
            "${CMAKE_CXX_FLAGS_RELEASE} /O3 /DNDEBUG /MD /arch:AVX2 /Gy /Gw /EHsc /fp:precise"
    )

    set(CMAKE_EXE_LINKER_FLAGS "/OPT:REF /OPT:ICF /INCREMENTAL:NO")
    set(CMAKE_SHARED_LINKER_FLAGS "/OPT:REF /OPT:ICF /INCREMENTAL:NO")
    set(CMAKE_MODULE_LINKER_FLAGS "/OPT:REF /OPT:ICF /INCREMENTAL:NO")
    set(CMAKE_STATIC_LINKER_FLAGS "")

elseif(WIN32)
    set(CMAKE_C_FLAGS_RELEASE
            "${CMAKE_C_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG /MD /arch:AVX2 /Gy /Gw /fp:precise"
    )
    set(CMAKE_CXX_FLAGS_RELEASE
            "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG /MD /arch:AVX2 /Gy /Gw /EHsc /fp:precise"
    )

    set(CMAKE_EXE_LINKER_FLAGS "/OPT:REF /OPT:ICF /INCREMENTAL:NO")
    set(CMAKE_SHARED_LINKER_FLAGS "/OPT:REF /OPT:ICF /INCREMENTAL:NO")
    set(CMAKE_MODULE_LINKER_FLAGS "/OPT:REF /OPT:ICF /INCREMENTAL:NO")
    set(CMAKE_STATIC_LINKER_FLAGS "")
endif()
