//
//  FunctionsTest.cpp
//  Cynapse
//
//

#include <Eigen/Dense>

#include "catch.hpp"
#include "../Functions.hpp"


TEST_CASE("Functions sigmoid", "[Functions]") {
    REQUIRE(sigmoid(1000) == 1.0);
    REQUIRE(sigmoid(0.0) == 0.5);
    REQUIRE(sigmoid(-1000) == 0.0);
}

TEST_CASE("Functions sigmoid_derivative", "[Functions]") {
    REQUIRE(sigmoid_derivative(1000) == 0.0);
    REQUIRE(sigmoid_derivative(0.0) == 0.25);
    REQUIRE(sigmoid_derivative(-1000) == 0.0);
}

TEST_CASE("Functions heaviside", "[Functions]") {
    REQUIRE(heaviside(0.49) == 0.0);
    REQUIRE(heaviside(0.5) == 1.0);
    REQUIRE(heaviside(0.51) == 1.0);
}

TEST_CASE("Functions quadratic_cost_derivative", "[Functions]") {
    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = 4;
    
    Eigen::MatrixXd n(2,2);
    m(0,0) = 10;
    m(1,0) = -3.5;
    m(0,1) = -6;
    m(1,1) = 2;
    
    
    Eigen::MatrixXd result = m - n;
    REQUIRE(quadratic_cost_derivative(m, n) == result);
}
