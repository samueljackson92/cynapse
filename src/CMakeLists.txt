cmake_minimum_required(VERSION 3.1)
project(Cynapse)

set(CMAKE_BUILD_TYPE Debug)
set(CYNAPSE_LIBRARY_NAME cynapse)
set(CYNAPSE_TOOLS_LIBRARY_NAME cynapse_tools)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

find_package(Eigen3 3.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories(${EIGEN3_LIBRARIES})

enable_testing()
add_subdirectory(core)
add_subdirectory(tools)
add_subdirectory(models)
add_subdirectory(test)
