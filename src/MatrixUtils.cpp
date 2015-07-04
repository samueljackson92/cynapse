
#include <Eigen/Dense>

#include <random>
#include <vector>

#include "Types.h"
#include "MatrixUtils.h"

void MatrixUtils::initializeRandomWeights(MatrixXd_ptr matrix) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i=0; i < matrix->cols(); ++i) {
        for (int j=0; j < matrix->rows(); ++j) {
            (*matrix)(j, i) = distribution(generator);
        }
    }
}

void MatrixUtils::applyFunction(MatrixXd_ptr matrix,
    std::function<double(double)> func) {
    for (int i=0; i < matrix->cols(); ++i) {
        for (int j=0; j < matrix->rows(); ++j) {
            (*matrix)(j, i) = func((*matrix)(j, i));
        }
    }
}
