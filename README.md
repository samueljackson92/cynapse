# Cynapse
C++ neural network library

## Bulding From Source

This project requires a C++11 compatible compiler and the following dependencies:

 - Eigen3
 - Boost
 - OpenCV

This project uses CMake. To build run the following: 

```bash
mkdir build
cd build
cmake ../src
make
```

## Running Unit Tests

We're using CTest, so all of the tests can be run simply by:
```bash
ctest
```
