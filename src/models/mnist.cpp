//
// Created by Samuel Jackson on 28/10/2015.
//


#include <boost/lexical_cast.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "mnist.h"
#include "../core/NeuralNetwork.h"
#include "../tools/BatchImageLoader.h"

const size_t BATCH_SIZE = 5;


size_t labelFromFilename(const std::string& filename) {
    size_t index = filename.find_last_of("_");
    return boost::lexical_cast<int>(filename.substr(index+1, 1));
}

Eigen::MatrixXd convertFilenamesToLabels(const std::vector<std::string>& filenames) {
    Eigen::MatrixXd mat(10, filenames.size());
    int i = 0;

    for (auto it = filenames.cbegin(); it != filenames.cend(); ++i, ++it) {
        size_t label = labelFromFilename(*it);
        Eigen::VectorXd v = Eigen::VectorXd::Zero(10);
        v(label) = 1;
        mat.col(i) = v;
    }

    return mat;
}


int main(int argc, char** argv )
{
    BatchImageLoader loader(argv[1], BATCH_SIZE);
    Eigen::MatrixXd examples = loader.next();
    std::vector<std::string> filenames = loader.getFilenames();

    Eigen::MatrixXd labels = convertFilenamesToLabels(filenames);
    std::cout << labels << std::endl;

    int nodes[3] = { static_cast<int>(examples.cols()), 50, 10 };
    std::vector<int> layout(&nodes[0], &nodes[0]+3);

    NeuralNetwork ann(layout, "sigmoid", false);
    ann.train(examples, labels, 500, 1.0);

    // Eigen::MatrixXd output = ann.feedForward(examples.col(0));
    //
    // std::cout << "========= Final Weights ========" << std::endl;
    // std::cout << output << std::endl;
    return 0;
}
