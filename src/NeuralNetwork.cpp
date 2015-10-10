
#include <Eigen/Dense>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "Functions.hpp"
#include "Types.h"
#include "NeuralNetwork.h"
#include "MatrixUtils.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

NeuralNetwork::NeuralNetwork(const std::vector<int>& layout,
    const std::string& funcName, const bool randomSeed) : m_layout(layout) {
    createLayers(randomSeed);
    createActivationFunction(funcName);
}

void NeuralNetwork::createLayers(const bool randomSeed) {
    // create layers. Ignore the first as this is the size of the input.
    m_weights.reserve(m_layout.size()-1);

    for (int i=1; i < m_layout.size(); ++i) {
        // size of matrix is defined by the number of nodes (rows) and the
        // size of the output vector from the previous layer (cols)
        Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(m_layout[i-1], m_layout[i]);
        // initialise matrix with random numbers drawn from Gaussian with
        // zero mean and unit variance
        MatrixUtils::initializeRandomWeights(weights, randomSeed);

        m_weights.push_back(weights);
    }
}

void NeuralNetwork::createActivationFunction(const std::string& funcName) {
    auto functions = FunctionFactory::create(funcName);
    m_activation_func = functions.first;
    m_activation_deriv = functions.second;
}

void NeuralNetwork::train(MatrixXd input, MatrixXd expected,
    int maxIter, double alpha) {

    for (int i=0; i <= maxIter; ++i) {

        MatrixXd output = feedForward(input);
        auto errors = backPropagate(output, expected);
        updateWeights(errors, alpha);

        if (i % 100 == 0) {
            std::cout << "Epoch " << i << ": "
                      << output.transpose() << std::endl;
        }
    }

}

MatrixXd NeuralNetwork::feedForward(MatrixXd input) {
    // Remove cached vectors from previous run
    m_zVectors.clear();
    m_activations.clear();

    MatrixXd activation = input;
    m_activations.push_back(activation);

    for (int index=0; index < m_layout.size()-1; ++index) {
        // compute weighted input for this layer
        MatrixXd weights = m_weights[index];
        MatrixXd z = weights.transpose() * activation;
        // cache result of weight-input combination for backprop
        m_zVectors.push_back(z);
        // compute the activation of function from the weighted input
        activation = z.unaryExpr(m_activation_func);
        // cache activation result of the weight-input combination for backprop
        m_activations.push_back(activation);
    }

    m_activations.pop_back();  // remove last activation

    return activation.transpose();
}

std::vector<MatrixXd> NeuralNetwork::backPropagate(MatrixXd output,
    MatrixXd actual) {

    if (checkIfMatriciesSizeMatch(output, actual)) {
        std::string msg = "Output and actual vectors must match in size.";
        throw std::runtime_error(msg);
    }

    if (checkIfFeedForwardPerformed()) {
        std::string msg = "Backpropagation may only be run after feed forward.";
        throw std::runtime_error(msg);
    }

    std::vector<MatrixXd> errors;

    // compute error for last layer of network
    MatrixXd cost = quadratic_cost_derivative(output, actual);
    MatrixXd sigmaPrime = getSigmaPrime();
    MatrixXd a = getActivation();

    MatrixXd delta = cost.transpose().cwiseProduct(sigmaPrime);
    MatrixXd wError = delta * a.transpose();

    errors.push_back(wError);

    // compute error in previous layers of network
    for (int index = m_weights.size()-2; index >= 0; --index) {
        MatrixXd weights = m_weights[index+1];
        sigmaPrime = getSigmaPrime();
        a = getActivation();

        delta = (weights * delta).cwiseProduct(sigmaPrime);
        wError = delta * a.transpose();

        errors.push_back(wError);
    }

    std::reverse(errors.begin(), errors.end());

    return std::vector<MatrixXd>(errors);
}

void NeuralNetwork::updateWeights(const std::vector<Eigen::MatrixXd>& errors,
    double alpha) {

    for (int i=0; i < m_weights.size(); ++i) {
        MatrixXd wError = errors[i].transpose();
        m_weights[i] = m_weights[i] - alpha * wError;
    }
}

Eigen::MatrixXd NeuralNetwork::getSigmaPrime() {
    if (m_zVectors.empty()) {
        std::runtime_error("Z value cache is empty.");
    }

    MatrixXd z = m_zVectors.back();
    m_zVectors.pop_back();  // remove z vector from cache
    return z.unaryExpr(m_activation_deriv);
}

Eigen::MatrixXd NeuralNetwork::getActivation() {
    if (m_activations.empty()) {
        std::runtime_error("Activation cache is empty.");
    }

    MatrixXd a = m_activations.back();
    m_activations.pop_back();
    return a;
}

void NeuralNetwork::printWeights() {
    for (int i =0; i < m_weights.size(); ++i) {
        std::cout << "Layer " << i << "------------------------" << std::endl;
        std::cout << m_weights[i] << std::endl;
    }
    std::cout << "-------------------------------" << std::endl;
}
