
#include <Eigen/Dense>

#include <string>
#include <vector>

#include "Types.h"

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

class NeuralNetwork {
 public:
     explicit NeuralNetwork(const std::vector<int>&, const std::string&,
         const bool randomSeed = true);
     Eigen::VectorXd feedForward(Eigen::VectorXd);
     std::vector<VectorXd_ptr> backPropagate(Eigen::VectorXd, Eigen::VectorXd);

 private:
     const std::vector<int> m_layout;
     std::function<double(double)> m_activation_func;
     std::vector<MatrixXd_ptr > m_layers;
     std::vector<VectorXd_ptr > m_zVectors;
     std::vector<VectorXd_ptr > m_activations;

     void createLayers(const bool randomSeed);
     void createActivationFunction(const std::string&);
};

#endif  // NEURALNETWORK_H_
