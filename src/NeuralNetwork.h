
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <string>
#include <vector>

#include "Types.h"

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

class NeuralNetwork {
 public:
     explicit NeuralNetwork(const std::vector<int>&, const std::string&,
         const bool randomSeed = true);
     Eigen::MatrixXd feedForward(Eigen::MatrixXd);
     void backPropagate(Eigen::MatrixXd, Eigen::MatrixXd);

 private:
     const std::vector<int> m_layout;
     std::function<double(double)> m_activation_func;
     std::function<double(double)> m_activation_deriv;

     std::vector<Eigen::MatrixXd> m_layers;
     std::vector<Eigen::MatrixXd> m_zVectors;
     std::vector<Eigen::MatrixXd> m_activations;

     void createLayers(const bool randomSeed);
     void createActivationFunction(const std::string&);
};

#endif  // NEURALNETWORK_H_
