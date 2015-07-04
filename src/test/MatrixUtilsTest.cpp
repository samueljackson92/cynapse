
#include <Eigen/Dense>

#include "catch.hpp"
#include "../MatrixUtils.h"

// Dummy function for testing applyFunction
double addOne(double x) {
    return x+1;
}

TEST_CASE("MatrixUtils initializeRandomWeights", "[MatrixUtils]") {
    Eigen::MatrixXd m = Eigen::Matrix2d::Zero();
    MatrixXd_ptr mptr = std::make_shared<Eigen::MatrixXd>(m);

    SECTION("Test method initialise randomly") {
        MatrixUtils::initializeRandomWeights(mptr);
        REQUIRE(mptr->isZero(0) == false);
    }

    SECTION("Test method initialise no seed") {
        Eigen::MatrixXd m2 = Eigen::Matrix2d::Zero();
        MatrixXd_ptr m2ptr = std::make_shared<Eigen::MatrixXd>(m2);

        MatrixUtils::initializeRandomWeights(mptr, false);
        MatrixUtils::initializeRandomWeights(m2ptr, false);

        REQUIRE(mptr->isZero(0) == false);
        REQUIRE(m2ptr->isZero(0) == false);

        REQUIRE(*mptr == *m2ptr);
    }
}

TEST_CASE("MatrixUtils applyFunction", "[MatrixUtils]") {
    Eigen::MatrixXd m = Eigen::Matrix2d::Zero();
    MatrixXd_ptr mptr = std::make_shared<Eigen::MatrixXd>(m);

    SECTION("Test method initialise randomly") {
        MatrixUtils::applyFunction(mptr, std::function<double(double)>(addOne));
        REQUIRE(mptr->isZero(0) == false);
        REQUIRE(mptr->isOnes(0) == true);
    }
}
