//
//  Functions.h
//  Functor classes for activation and activation derivative functions.
//

#include <Eigen/Dense>
#include <cmath>

#ifndef Cynapse_Functions_h
#define Cynapse_Functions_h

double sigmoid(double theta) {
    return 1.0 / (1.0 + exp(-theta));
}

double sigmoid_derivative(double z) {
    return sigmoid(z) * (1-sigmoid(z));
}

double heaviside(double theta) {
    return (theta >= 0.5 ? 1.0 : 0.0);
}

Eigen::MatrixXd quadratic_cost_derivative(Eigen::MatrixXd output, Eigen::MatrixXd input) {
    return output - input;
}

#endif
