
#include <Eigen/Dense>

#include <iostream>
#include <cmath>
#include <string>
#include <vector>

#include "Types.h"
#include "NeuralNetwork.h"
#include "MatrixUtils.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

double sigmoid_activation(double theta) {
    return 1.0 / (1.0 + exp(-theta));
}

double step_activation(double theta) {
    return (theta >= 0.5 ? 1.0 : 0.0);
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layout,
    const std::string& funcName, const bool randomSeed) : m_layout(layout) {
    createLayers(randomSeed);
    createActivationFunction(funcName);
}

void NeuralNetwork::createLayers(const bool randomSeed) {
    // create layers. Ignore the first as this is the size of the input.
    m_layers.reserve(m_layout.size()-1);
    for (int i=1; i < m_layout.size(); ++i) {
        // size of matrix is defined by the number of nodes (rows) and the
        // size of the output vector from the previous layer (cols)
        Eigen::MatrixXd layer(m_layout[i], m_layout[i-1]);
        MatrixXd_ptr layer_ptr = std::make_shared<Eigen::MatrixXd>(layer);
        // initialise matrix with random numbers drawn from Gaussian with
        // zero mean and unit variance
        MatrixUtils::initializeRandomWeights(layer_ptr, randomSeed);
        m_layers.push_back(layer_ptr);
    }
}

void NeuralNetwork::createActivationFunction(const std::string& funcName) {
    if (funcName == "sigmoid") {
        m_activation_func = std::function<double(double)>(sigmoid_activation);
    } else if (funcName == "step") {
        m_activation_func = std::function<double(double)>(step_activation);
    } else {
        throw std::runtime_error("Function " + funcName + " is not supported.");
    }
}

VectorXd NeuralNetwork::feedForward(VectorXd inputVec) {
    MatrixXd_ptr input = std::make_shared<MatrixXd>(inputVec);
    MatrixXd_ptr output;

    for (auto layerIt = m_layers.begin(); layerIt != m_layers.end(); ++layerIt) {
        // compute activation for layer
        output = std::make_shared<MatrixXd>((**layerIt) * (*input));
        MatrixUtils::applyFunction(output, m_activation_func);
        // set the output as the input for the next layer
        input = output;
    }

    return *output;
}
