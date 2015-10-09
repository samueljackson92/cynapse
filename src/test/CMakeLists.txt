
set(TEST_INCLUDES
    catch.hpp
    TestsCatchRequire.h
)

set(ALL_TEST_FILES
    NeuralNetworkTest.cpp
    MatrixUtilsTest.cpp
)

foreach(TEST_FILE ${ALL_TEST_FILES})
    # Get name of file without extension
    get_filename_component(TEST_EXEC_NAME ${TEST_FILE} NAME_WE)

    # Create executable for test case
    set(TEST_FILES ${TEST_INCLUDES} ${TEST_FILE})
    add_executable(${TEST_EXEC_NAME} ${TEST_FILES})
    target_link_libraries(${TEST_EXEC_NAME} ${CYNAPSE_LIBRARY_NAME} ${EIGEN3_LIBRARIES})
    set_property(TARGET ${TEST_EXEC_NAME} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${TEST_EXEC_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

    # Add the test to CUnit
    add_test (NAME ${TEST_EXEC_NAME} COMMAND ${TEST_EXEC_NAME})
endforeach()
