
#include <Eigen/Dense>

#include "TestsCatchRequire.h"
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
