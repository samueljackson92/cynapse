
set(MNIST_MODEL cynapse_models_mnist)
add_executable(${MNIST_MODEL} mnist.cpp mnist.h)
target_link_libraries(${MNIST_MODEL} ${CYNAPSE_LIBRARY_NAME} ${CYNAPSE_TOOLS_LIBRARY_NAME})
set_property(TARGET ${MNIST_MODEL} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${MNIST_MODEL} PROPERTY CXX_STANDARD_REQUIRED ON)
