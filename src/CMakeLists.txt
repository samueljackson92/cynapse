cmake_minimum_required (VERSION 3.1)
project (ANN)

set(CMAKE_BUILD_TYPE Debug)

set(EXEC_NAME ann)
set(TEST_EXEC_NAME all_tests)

set(TEST_DIR test)

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

add_library(neural_net STATIC ${INCLUDE_FILES} ${SRC_FILES})

add_executable(${TEST_EXEC_NAME} ${TEST_FILES})
target_link_libraries(${TEST_EXEC_NAME} neural_net)
set_property(TARGET ${TEST_EXEC_NAME} PROPERTY CXX_STANDARD_REQUIRED 11)
