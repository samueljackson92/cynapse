cmake_minimum_required(VERSION 3.1)
project(Cynapse)

set(CMAKE_BUILD_TYPE Debug)
set(CYNAPSE_LIBRARY_NAME cynapse)
set(TEST_EXEC_NAME all_tests)
set(TEST_DIR test)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

find_package(Eigen3 3.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories(${EIGEN3_LIBRARIES})

set(INC_FILES
    NeuralNetwork.h
    MatrixUtils.h
    Types.h
)

set(SRC_FILES
    NeuralNetwork.cpp
    MatrixUtils.cpp
)

set(TEST_FILES
    ${TEST_DIR}/catch.hpp
    ${TEST_DIR}/TestsCatchRequire.h
    ${TEST_DIR}/NeuralNetworkTest.cpp
    ${TEST_DIR}/MatrixUtilsTest.cpp
)

add_library(${CYNAPSE_LIBRARY_NAME} STATIC ${INCLUDE_FILES} ${SRC_FILES})
target_link_libraries(${CYNAPSE_LIBRARY_NAME} ${EIGEN3_LIBRARIES})
set_property(TARGET ${CYNAPSE_LIBRARY_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${CYNAPSE_LIBRARY_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(${TEST_EXEC_NAME} ${TEST_FILES})
target_link_libraries(${TEST_EXEC_NAME} ${CYNAPSE_LIBRARY_NAME} ${EIGEN3_LIBRARIES})
set_property(TARGET ${TEST_EXEC_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${TEST_EXEC_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
