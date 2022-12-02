cmake_minimum_required(VERSION 3.2)
project(shadowbox)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(NPE_PYTHON_VERSION 3.10)
set(USE_BLOSC OFF)
set(USE_ZLIB OFF)
set(OPENVDB_BUILD_VDB_PRINT OFF)
# set(OPENVDB_CORE_SHARED OFF)
# set(OPENVDB_BUILD_STATIC_LIBS ON)
# option(FETCHCONTENT_FULLY_DISCONNECTED "" ON)
# option(UPDATE_DISCONNECTED "" OFF)

if(WIN32)
    set(BOOST_ROOT "C:/Boost")
    set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include/boost-1_80")
    set(TBB_ROOT "C:/Tbb")
    set(TBB_LIBRARYDIR "${TBB_ROOT}/lib/intel64/vc14")
    set(Boost_USE_STATIC_LIBS OFF)
endif()

include(FetchContent)
FetchContent_Declare(numpyeigen
    GIT_REPOSITORY https://github.com/fwilliams/numpyeigen.git
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/external/numpyeigen"
    GIT_TAG 80bdcef3d6dc891c275475f28437f17f7deebc47
    )
FetchContent_Declare(openvdb
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openvdb
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/external/openvdb"
    GIT_TAG 65a39c880769da171261b5d00b2e350a8c015b26
    )

FetchContent_GetProperties(numpyeigen)
FetchContent_GetProperties(openvdb)

if(NOT numpyeigen_POPULATED)
    FetchContent_Populate(numpyeigen)
    list(APPEND CMAKE_MODULE_PATH ${numpyeigen_SOURCE_DIR}/cmake)
    include(numpyeigen)
endif()

if(NOT openvdb_POPULATED)
    FetchContent_Populate(openvdb)
    add_subdirectory(${openvdb_SOURCE_DIR} ${openvdb_BINARY_DIR})
endif()

# This creates a CMake target representing an importable python module.
# The first argument is the name of the target (in this case `npe_example`).
# It is also the name of the Python module (i.e. `import npe_example` in Python)
# The files specified after BINDING_SOURCES are the source files containing numpyeigen bindings
npe_add_module(core
    BINDING_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/create_mesh.cpp
    )

# Use SSE2 instructions
# set_target_properties(core PROPERTIES COMPILE_FLAGS "-fvisibility=hidden -msse2")

if(WIN32)
    target_include_directories(openvdb_static PUBLIC
        ${BOOST_INCLUDEDIR}
        )
endif()

target_link_libraries(core PRIVATE openvdb_static)
