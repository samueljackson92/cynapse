
#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

using Eigen::Vector2d;
using Eigen::VectorXd;

int main(int argc, char** argv) {

    // two layers each with two nodes.
    int nodes[3] = { 2, 2, 1};
    std::vector<int> layout(&nodes[0], &nodes[0]+3);

    NeuralNetwork ann(layout, "step");
    std::cout << "Neural Network ----------------------------" << std::endl;

    Vector2d input(1, 1);
    VectorXd result = ann.feedForward(input);

    std::cout << result << std::endl;

    return 0;
}
