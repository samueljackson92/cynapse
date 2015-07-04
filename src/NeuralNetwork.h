
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

 private:
     const std::vector<int> m_layout;
     std::function<double(double)> m_activation_func;
     std::vector<MatrixXd_ptr > m_layers;

     void createLayers(const bool randomSeed);
     void createActivationFunction(const std::string&);
};

#endif  // NEURALNETWORK_H_
