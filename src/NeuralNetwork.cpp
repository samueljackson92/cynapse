
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

MatrixXd cost_derivative(MatrixXd output, MatrixXd input) {
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
        // initialise matrix with random numbers drawn from Gaussian with
        // zero mean and unit variance
        MatrixUtils::initializeRandomWeights(layer, randomSeed);
        m_layers.push_back(layer);
    }
}

void NeuralNetwork::createActivationFunction(const std::string& funcName) {
    if (funcName == "sigmoid") {
        m_activation_func = std::function<double(double)>(sigmoid_activation);
        m_activation_deriv = std::function<double(double)>(sigmoid_prime);
    } else if (funcName == "step") {
        m_activation_func = std::function<double(double)>(step_activation);
    } else {
        throw std::runtime_error("Function " + funcName + " is not supported.");
    }
}

MatrixXd NeuralNetwork::feedForward(MatrixXd input) {
    m_zVectors.clear();  // Remove cached vectors from previous run
    m_activations.push_back(input);

    MatrixXd output;

    for (auto layerIt = m_layers.begin(); layerIt != m_layers.end(); ++layerIt) {
        // compute weighted input for this layer
        MatrixXd weights = *layerIt;
        output = weights * input;
        // cache result of weight-input combination for backprop
        m_zVectors.push_back(output);
        // compute the activation of function from the weighted input
        MatrixUtils::applyFunction(output, m_activation_func);
        // cache activation result of the weight-input combination for backprop
        m_activations.push_back(output);
        // set the output as the input for the next layer
        input = output;
    }

    m_activations.pop_back();  // remove last uneeded activation

    return output;
}

 void NeuralNetwork::backPropagate(MatrixXd output, MatrixXd actual) {
    double alpha = 0.001;

    if (output.size() != actual.size()) {
        throw std::runtime_error("Output and actual vectors must match in size!");
    }

    MatrixXd weights = m_layers.back();
    // compute error for last layer of network
    MatrixXd cost = cost_derivative(output, actual);
    MatrixXd z = m_zVectors.back();
    m_zVectors.pop_back();  // remove z vector from cache

    MatrixXd sigma = z.unaryExpr(m_activation_deriv);
    MatrixXd delta = cost.cwiseProduct(sigma);

    MatrixXd gradient = m_activations.back() * delta;
    m_activations.pop_back();

    weights = weights - (alpha * gradient).transpose();

    // compute error in previous layers of network
    //for each layer:
    for (int index = m_layers.size()-2; index > 0; --index) {
        weights = m_layers[index];

        z = m_zVectors.back();
        m_zVectors.pop_back();

        sigma = m_zVectors.back().unaryExpr(m_activation_deriv);
        delta = (weights.transpose() * delta).cwiseProduct(sigma);

        gradient = m_activations.back() * delta;
        m_activations.pop_back();

        weights = m_layers[index];
        weights = weights - alpha * gradient;
    }
}
