cmake_minimum_required(VERSION 3.15)
project(shadowbox)

if(NOT "${CMAKE_CXX_STANDARD}")
  set(CMAKE_CXX_STANDARD 17)
endif()

option(OPENVDB_USE_STATIC_LIBS "" ON)

if(MSVC)
    list(APPEND CMAKE_MODULE_PATH "C:/Program Files/OpenVDB/lib/cmake/OpenVDB")
    set(pybind11_DIR "C:/Program Files/Python310/Lib/site-packages/pybind11/share/cmake/pybind11")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenVDB REQUIRED)

pybind11_add_module(core
    src/create_mesh.cpp
    )

set_target_properties(core PROPERTIES
    MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<NOT:$<BOOL:${OPENVDB_USE_STATIC_LIBS}>>:DLL>"
    )

target_link_libraries(core PRIVATE OpenVDB::openvdb)
