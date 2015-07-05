
#include <Eigen/Dense>

#include <random>
#include <vector>

#include "MatrixUtils.h"

void MatrixUtils::initializeRandomWeights(Eigen::MatrixXd& matrix,
    const bool randomSeed) {

    unsigned seed = 0;
    if (randomSeed) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }

    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i=0; i < matrix.cols(); ++i) {
        for (int j=0; j < matrix.rows(); ++j) {
            matrix(j, i) = distribution(generator);
        }
    }
}
