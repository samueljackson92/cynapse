
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

double sigmoid_prime(double z) {
    return sigmoid_activation(z) * (1-sigmoid_activation(z));
}

VectorXd cost_derivative(VectorXd output, VectorXd input) {
    return output - input;
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
        Eigen::MatrixXd layer = Eigen::MatrixXd::Zero(m_layout[i], m_layout[i-1]);
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

VectorXd NeuralNetwork::feedForward(VectorXd input) {
    m_zVectors.clear();  // Remove cached vectors from previous run
    m_activations.push_back(std::make_shared<VectorXd>(input));

    VectorXd output;

    for (auto layerIt = m_layers.begin(); layerIt != m_layers.end(); ++layerIt) {
        // compute weighted input for this layer
        MatrixXd weightMatrix = **layerIt;
        output = weightMatrix * input;
        // cache result of weight-input combination for backprop
        m_zVectors.push_back(std::make_shared<VectorXd>(output));
        // compute the activation of function from the weighted input
        MatrixUtils::applyFunction(output, m_activation_func);
        // cache activation result of the weight-input combination for backprop
        m_activations.push_back(std::make_shared<VectorXd>(output));
        // set the output as the input for the next layer
        input = output;
    }

    return output;
}

 std::vector<VectorXd_ptr> NeuralNetwork::backPropagate(VectorXd output, VectorXd actual) {
    std::vector<VectorXd_ptr> gradients;
    m_activations.pop_back();

    if (output.size() != actual.size()) {
        throw std::runtime_error("Output and actual vectors must match in size!");
    }

    auto activation_deriv = std::function<double(double)>(sigmoid_prime);

    // compute error for last layer of network
    VectorXd cost = cost_derivative(output, actual);
    VectorXd sigma = m_zVectors.back()->unaryExpr(activation_deriv);
    m_zVectors.pop_back();
    VectorXd delta = cost.cwiseProduct(sigma);

    VectorXd gradient = m_activations.back() * delta;
    m_activations.pop_back();
    gradients.push_back(std::make_shared<VectorXd>(gradient));

    // compute error in previous layers of network
    //for each layer:
    for (int index = m_layers.size()-1; index > 0; --index) {
        MatrixXd weightMatrix = *m_layers[index];
        sigma = m_zVectors.back()->unaryExpr(activation_deriv);
        m_zVectors.pop_back();
        delta = (weightMatrix.transpose() * delta).cwiseProduct(sigma);

        gradient = m_activations.back() * delta;
        m_activations.pop_back();
        gradients.push_back(std::make_shared<VectorXd>(gradient));
    }

    return gradients;
}
