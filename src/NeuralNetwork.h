
#include <Eigen/Dense>

#include <string>
#include <vector>

#include "Types.h"

class NeuralNetwork {
 public:
     explicit NeuralNetwork(const std::vector<int>&, const std::string&);
     Eigen::VectorXd feedForward(Eigen::VectorXd);

 private:
     const std::vector<int> m_layout;
     std::function<double(double)> m_activation_func;
     std::vector<MatrixXd_ptr > m_layers;

     void createLayers();
     void createActivationFunction(const std::string&);
};
