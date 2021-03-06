
#include <Eigen/Dense>
#include <vector>

#include "TestsCatchRequire.h"
#include "../core/NeuralNetwork.h"

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
        int nodes[3] = { 3, 2, 1};
        std::vector<int> layout(&nodes[0], &nodes[0]+3);

        NeuralNetwork ann(layout, "step", false);

        Eigen::MatrixXd input(1, 3);
        input << 1, 1, 1;

        Eigen::MatrixXd output = ann.feedForward(input);

        REQUIRE(output.value() == 0);
    }

    SECTION("Check feedForward with multiple outputs") {
        int nodes[3] = { 3, 2, 2};
        std::vector<int> layout(&nodes[0], &nodes[0]+3);

        NeuralNetwork ann(layout, "sigmoid", false);

        Eigen::MatrixXd input(1, 3);
        input << 1, 1, 1;
        
        Eigen::MatrixXd output = ann.feedForward(input);

        REQUIRE(output(0, 0) == Approx(0.276810));
        REQUIRE(output(1, 0) == Approx(0.757047));
    }

    SECTION("Check feedForward single perceptron") {
        int nodes[2] = { 2, 1};
        std::vector<int> layout(&nodes[0], &nodes[0]+2);

        NeuralNetwork ann(layout, "sigmoid", false);

        Eigen::MatrixXd input(1, 2);
        input << 1, 1;

        Eigen::MatrixXd output = ann.feedForward(input);
        REQUIRE(output.value() == Approx(0.16148));
    }

    SECTION("Check feedForward with multiple inputs", "[NeuralNetwork]") {
        int nodes[3] = { 2, 2, 1 };
        std::vector<int> layout(&nodes[0], &nodes[0]+3);

        NeuralNetwork ann(layout, "sigmoid", false);

        Eigen::MatrixXd input(4, 2);
        input << 1, 1,
                 0, 1,
                 1, 0,
                 0, 0;

        Eigen::MatrixXd output = ann.feedForward(input);
        REQUIRE(output.size() == 4);

        Eigen::MatrixXd expectedResult(1, 4);
        expectedResult << 0.36363, 0.323174, 0.350235, 0.30499;
        REQUIRE(output.size() == expectedResult.size());

        for (int i=0; i < output.cols(); ++i) {
            for (int j=0; j < output.rows(); ++j) {
                REQUIRE(output(j, i) == Approx(expectedResult(j, i)));
            }
        }

    }


}

TEST_CASE("Testing NeuralNetwork backPropagate", "[NeuralNetwork]") {
    // create a basic layout 2 inputs, 2 nodes in the 1st layer 1 in the 2nd

    SECTION("Check backprop with normal input", "[NeuralNetwork]") {
        int nodes[3] = { 3, 2, 1};
        std::vector<int> layout(&nodes[0], &nodes[0]+3);

        NeuralNetwork ann(layout, "sigmoid", false);

        Eigen::MatrixXd input(1, 3);
        input << 1, 1, 1;
        
        Eigen::VectorXd actual(1);
        actual << 1;

        Eigen::MatrixXd output = ann.feedForward(input);

        REQUIRE(output.size() == 1);
        REQUIRE(output.value() == Approx(0.27681));

        REQUIRE_NOTHROW(ann.backPropagate(output, actual));
    }

    SECTION("Check backprop with two outputs", "[NeuralNetwork]") {
        int nodes[3] = { 3, 2, 2};
        std::vector<int> layout(&nodes[0], &nodes[0]+3);

        NeuralNetwork ann(layout, "sigmoid", false);

        Eigen::MatrixXd input(1, 3);
        input << 1, 1, 1;
        
        Eigen::MatrixXd output = ann.feedForward(input);

        REQUIRE(output.cols() == 1);
        REQUIRE(output.rows() == 2);

        REQUIRE(output(0, 0) == Approx(0.276810));
        REQUIRE(output(1, 0) == Approx(0.757047));

        Eigen::MatrixXd actual(2, 1);
        actual << 1, 1;

        REQUIRE_NOTHROW(ann.backPropagate(output, actual));
    }
}

TEST_CASE("Testing NeuralNetwork multiple inputs", "[NeuralNetwork]") {
    int nodes[3] = { 3, 3, 1 };
    std::vector<int> layout(&nodes[0], &nodes[0]+3);
    NeuralNetwork ann(layout, "sigmoid", false);

    Eigen::MatrixXd input(4, 3);
    input << 1, 1, 1,
             1, 1, 0,
             1, 0, 1,
             1, 0, 0;

    Eigen::MatrixXd output = ann.feedForward(input);
    REQUIRE(output.cols() == 4);
    REQUIRE(output.rows() == 1);

    Eigen::MatrixXd expectedResult(1, 4);
    expectedResult << 0.39573, 0.64993, 0.29571, 0.48559;

    REQUIRE(output.cols() == expectedResult.cols());
    REQUIRE(output.rows() == expectedResult.rows());

    for (int i=0; i < output.cols(); ++i) {
        for (int j=0; j < output.rows(); ++j) {
            REQUIRE(output(j, i) == Approx(expectedResult(j, i)));
        }
    }

    Eigen::MatrixXd actual(1, 4);
    actual << 0, 1, 1, 0;

    REQUIRE_NOTHROW(ann.backPropagate(output, actual));
}

TEST_CASE("Testing NeuralNetwork training", "[NeuralNetwork]") {
    int nodes[3] = { 3, 3, 1 };
    std::vector<int> layout(&nodes[0], &nodes[0]+3);

    NeuralNetwork ann(layout, "sigmoid", false);

    Eigen::MatrixXd input(4, 3);
    input << 1, 1, 1,
             1, 1, 0,
             1, 0, 1,
             1, 0, 0;

    Eigen::MatrixXd actual(1, 4);
    actual << 0, 1, 1, 0;

    REQUIRE_NOTHROW(ann.train(input, actual, 5000, 1));

    Eigen::MatrixXd example(1, 3);
    example << 1, 0, 0;

    Eigen::MatrixXd output = ann.feedForward(example);
    REQUIRE(output.value() == Approx(0.0328009245));
}
