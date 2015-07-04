cmake_minimum_required (VERSION 3.1)
project (ANN)

# set(TEST_DIR test)

set(INC_FILES
    NeuralNetwork.h
    MatrixUtils.h
    Types.h
)

set(SRC_FILES
    main.cpp
    NeuralNetwork.cpp
    MatrixUtils.cpp
)

# set(TEST_FILES
#     ${TEST_DIR}/NeuralNetworkTest.cpp
#     ${TEST_DIR}/MatrixUtilsTest.cpp
# )

add_executable(ann ${INCLUDE_FILES} ${SRC_FILES})
# add_executable(test_all ${INCLUDE_FILES} ${SRC_FILES} ${TEST_FILES})

set_property(TARGET ann PROPERTY CXX_STANDARD_REQUIRED 11)
