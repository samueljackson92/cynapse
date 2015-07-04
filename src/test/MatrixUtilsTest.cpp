
#include <Eigen/Dense>

#include "catch.hpp"
#include "../MatrixUtils.h"

// Dummy function for testing applyFunction
double addOne(double x) {
    return x+1;
}

TEST_CASE("MatrixUtils initializeRandomWeights", "[MatrixUtils]") {
    Eigen::MatrixXd m = Eigen::Matrix2d::Zero();

    SECTION("Test method initialise randomly") {
        MatrixUtils::initializeRandomWeights(m);
        REQUIRE(m.isZero(0) == false);
    }

    SECTION("Test method initialise no seed") {
        Eigen::MatrixXd m2 = Eigen::Matrix2d::Zero();

        MatrixUtils::initializeRandomWeights(m, false);
        MatrixUtils::initializeRandomWeights(m2, false);

        REQUIRE(m.isZero(0) == false);
        REQUIRE(m2.isZero(0) == false);

        REQUIRE(m == m2);
    }
}

TEST_CASE("MatrixUtils applyFunction", "[MatrixUtils]") {
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(3, 3);

    SECTION("Test method initialise randomly") {
        MatrixUtils::applyFunction(m, std::function<double(double)>(addOne));
        REQUIRE(m.isZero(0) == false);
        REQUIRE(m.isOnes(0) == true);
    }
}
