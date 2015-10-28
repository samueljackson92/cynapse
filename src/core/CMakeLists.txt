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


