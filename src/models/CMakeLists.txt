
find_package( OpenCV REQUIRED )
add_executable(mnist mnist.cpp mnist.h)
target_link_libraries(mnist ${OpenCV_LIBS} ${CYNAPSE_LIBRARY_NAME} ${EIGEN3_LIBRARIES})
set_property(TARGET mnist PROPERTY CXX_STANDARD 11)
set_property(TARGET mnist PROPERTY CXX_STANDARD_REQUIRED ON)

