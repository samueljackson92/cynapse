cmake_minimum_required(VERSION 3.1)
project(Cynapse)

set(CMAKE_BUILD_TYPE Debug)
set(CYNAPSE_LIBRARY_NAME cynapse)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

find_package(Eigen3 3.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories(${EIGEN3_LIBRARIES})

set(INC_FILES
    Functions.hpp
    NeuralNetwork.h
    MatrixUtils.h
    Types.h
)

set(SRC_FILES
    NeuralNetwork.cpp
    MatrixUtils.cpp
)

# Build the Cynapse library
add_library(${CYNAPSE_LIBRARY_NAME} STATIC ${INCLUDE_FILES} ${SRC_FILES})
target_link_libraries(${CYNAPSE_LIBRARY_NAME} ${EIGEN3_LIBRARIES})
set_property(TARGET ${CYNAPSE_LIBRARY_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${CYNAPSE_LIBRARY_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

# Build the tests
enable_testing()
add_subdirectory(test)
