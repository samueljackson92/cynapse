
#include <Eigen/Dense>
#include <vector>

#include "TestsCatchRequire.h"
#include "../NeuralNetwork.h"

TEST_CASE("NeuralNetwork can be initialised", "[NeuralNetwork]") {
    // create a basic layout 2 inputs, 2 nodes in the 1st layer 1 in the 2nd
    int nodes[3] = { 2, 2, 1};
    std::vector<int> layout(&nodes[0], &nodes[0]+3);

    SECTION("Creating a network with valid parameters") {
        REQUIRE_NOTHROW(NeuralNetwork(layout, "step"));
    }

    SECTION("Creating a network with sigmoid function") {
        REQUIRE_NOTHROW(NeuralNetwork(layout, "sigmoid"));
    }

    SECTION("Creating a network with invalid function") {
        REQUIRE_THROWS(NeuralNetwork(layout, "somefunc"));
    }

    SECTION("Creating a network without a random seed") {
        REQUIRE_NOTHROW(NeuralNetwork(layout, "step", false));
    }
}

TEST_CASE("Testing NeuralNetwork feedForward", "[NeuralNetwork]") {

    SECTION("Check feedForward with valid parameters") {
        int nodes[3] = { 2, 2, 1};
        std::vector<int> layout(&nodes[0], &nodes[0]+3);

        NeuralNetwork ann(layout, "step", false);

        Eigen::Vector2d input(1, 1);
        Eigen::VectorXd output = ann.feedForward(input);

        REQUIRE(output.size() == 1);
        REQUIRE(output.value() == 0);
    }

    SECTION("Check feedForward with multiple outputs") {
        int nodes[3] = { 3, 2, 2};
        std::vector<int> layout(&nodes[0], &nodes[0]+3);

        NeuralNetwork ann(layout, "sigmoid", false);

        Eigen::Vector3d input(1, 1, 1);
        Eigen::VectorXd output = ann.feedForward(input);

        REQUIRE(output.size() == 2);
        REQUIRE(output[0] == Approx(0.442010));
        REQUIRE(output[1] == Approx(0.464295));
    }
}

TEST_CASE("Testing NeuralNetwork backPropagate", "[NeuralNetwork]") {
    // create a basic layout 2 inputs, 2 nodes in the 1st layer 1 in the 2nd
    int nodes[3] = { 3, 2, 1};
    std::vector<int> layout(&nodes[0], &nodes[0]+3);

    NeuralNetwork ann(layout, "sigmoid", false);

    Eigen::Vector3d input(1, 1, 1);
    Eigen::VectorXd actual(1);
    actual << 1;
    Eigen::VectorXd output = ann.feedForward(input);
    ann.backPropagate(output, actual);
}
