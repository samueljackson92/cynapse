add_executable(mnist mnist.cpp mnist.h)
target_link_libraries(mnist ${CYNAPSE_LIBRARY_NAME} ${CYNAPSE_TOOLS_LIBRARY_NAME})
set_property(TARGET mnist PROPERTY CXX_STANDARD 11)
set_property(TARGET mnist PROPERTY CXX_STANDARD_REQUIRED ON)
