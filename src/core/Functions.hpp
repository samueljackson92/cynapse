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

class FunctionFactory
{
public:
    static std::pair<std::function<double(double)>, std::function<double(double)> >
        create(const std::string& funcName) {
        
        std::function<double(double)> activation_func;
        std::function<double(double)> activation_deriv;
        
        if (funcName == "sigmoid") {
            activation_func = std::function<double(double)>(sigmoid);
            activation_deriv = std::function<double(double)>(sigmoid_derivative);
        }
        else if (funcName == "step")
        {
            activation_func = std::function<double(double)>(heaviside);
        }
        else
        {
            throw std::runtime_error("Function " + funcName + " is not supported.");
        }
            
        return std::pair<std::function<double(double)>, std::function<double(double)> >(activation_func, activation_deriv);
    }
};

#endif
