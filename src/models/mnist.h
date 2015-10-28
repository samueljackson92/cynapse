//
// Created by Samuel Jackson on 28/10/2015.
//

#ifndef CYNAPSE_MNIST_CPP_H
#define CYNAPSE_MNIST_CPP_H

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

Eigen::VectorXf convert_image_to_vector(cv::Mat image);

#endif //CYNAPSE_MNIST_CPP_H
