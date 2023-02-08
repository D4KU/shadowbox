cmake_minimum_required(VERSION 3.15)
project(shadowbox)

if(NOT "${CMAKE_CXX_STANDARD}")
  set(CMAKE_CXX_STANDARD 17)
endif()

if(MSVC)
    option(OPENVDB_USE_STATIC_LIBS "" ON)
    file(TO_CMAKE_PATH "$ENV{ProgramFiles}" PROGRAM_FILES)
    list(APPEND CMAKE_MODULE_PATH "${PROGRAM_FILES}/OpenVDB/lib/cmake/OpenVDB")
else()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    list(APPEND CMAKE_MODULE_PATH "/usr/lib/x86_64-linux-gnu/cmake/OpenVDB")
endif()

find_package(Eigen3 CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenVDB REQUIRED)

pybind11_add_module(core
    src/create_mesh.cpp
    )

set_target_properties(core PROPERTIES
    MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<NOT:$<BOOL:${OPENVDB_USE_STATIC_LIBS}>>:DLL>"
    )

if(MSVC)
    set_target_properties(core PROPERTIES SUFFIX ".pyd")
endif()

target_link_libraries(core PRIVATE Eigen3::Eigen)
target_link_libraries(core PRIVATE OpenVDB::openvdb)
