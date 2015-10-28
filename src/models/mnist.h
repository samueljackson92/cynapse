//
// Created by Samuel Jackson on 28/10/2015.
//

#ifndef CYNAPSE_MNIST_CPP_H
#define CYNAPSE_MNIST_CPP_H

#include <Eigen/Dense>

size_t labelFromFilename(const std::string& filename);
Eigen::MatrixXd convertFilenamesToLabels(const std::vector<std::string>& filenames);

#endif //CYNAPSE_MNIST_CPP_H
