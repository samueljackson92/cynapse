//
// Created by Samuel Jackson on 28/10/2015.
//


#include <boost/lexical_cast.hpp>

#include <iostream>
#include <vector>

#include "mnist.h"
#include "../tools/BatchImageLoader.h"

const size_t BATCH_SIZE = 5;
const size_t NUM_CLASSES = 10;
const size_t INPUT_SIZE = 785;

int main(int argc, char** argv ) {
    NeuralNetwork ann = createNetwork();

    BatchImageLoader loader(argv[1], BATCH_SIZE, NUM_CLASSES);
    Eigen::MatrixXd examples;
    Eigen::MatrixXd labels;

    int batchCount = 0;
    while (loader.hasNext() && batchCount < 10) {
        std::cout << "BATCH " << batchCount << "===========================" << std::endl;

        loader.next();
        examples = loader.loadImages();
        labels = loader.loadLabels();

        ann.train(examples, labels, 1000, .1);
        ++batchCount;
    }

    Eigen::MatrixXd output = ann.feedForward(examples);

    std::cout << output << std::endl;
    std::cout << labels<< std::endl;


}

NeuralNetwork createNetwork() {
    int nodes[3] = {INPUT_SIZE, 50, NUM_CLASSES};
    std::vector<int> layout(&nodes[0], &nodes[0]+3);

    NeuralNetwork ann(layout, "sigmoid", false);
    return ann;
}