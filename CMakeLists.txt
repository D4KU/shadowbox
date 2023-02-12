cmake_minimum_required(VERSION 3.15)
project(shadowbox)

if(NOT "${CMAKE_CXX_STANDARD}")
  set(CMAKE_CXX_STANDARD 17)
endif()

option(OPENVDB_USE_STATIC_LIBS "" ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Eigen3 CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenVDB REQUIRED)

pybind11_add_module(core
    src/create_mesh.cpp
    )

if(MSVC)
    set_target_properties(core PROPERTIES
        SUFFIX ".pyd"
        MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<NOT:$<BOOL:${OPENVDB_USE_STATIC_LIBS}>>:DLL>")
endif()

target_link_libraries(core PRIVATE Eigen3::Eigen)
target_link_libraries(core PRIVATE OpenVDB::openvdb)
