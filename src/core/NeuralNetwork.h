
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

     void train(Eigen::MatrixXd, Eigen::MatrixXd,
        int maxIter = 500, double alpha = 0.01);

     std::vector<Eigen::MatrixXd>
        backPropagate(Eigen::MatrixXd, Eigen::MatrixXd);

     void updateWeights(const std::vector<Eigen::MatrixXd>&, double);

    void printWeights();

 private:
     const std::vector<int> m_layout;
     std::function<double(double)> m_activation_func;
     std::function<double(double)> m_activation_deriv;

     std::vector<Eigen::MatrixXd> m_weights;
     std::vector<Eigen::MatrixXd> m_zVectors;
     std::vector<Eigen::MatrixXd> m_activations;


     void createLayers(const bool randomSeed);
     void createActivationFunction(const std::string&);

     Eigen::MatrixXd getSigmaPrime();
     Eigen::MatrixXd getActivation();

     bool checkIfFeedForwardPerformed() {
         return (m_activations.size() == 0 || m_zVectors.size() == 0);
     }

     bool checkIfMatriciesSizeMatch(Eigen::MatrixXd lhs, Eigen::MatrixXd rhs) {
         return (lhs.size() != rhs.size());
     }

};

#endif  // NEURALNETWORK_H_
