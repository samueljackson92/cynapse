
#include <Eigen/Dense>
#include <vector>

#include "Types.h"

#ifndef MATRIXUTILS_H_
#define MATRIXUTILS_H_

class MatrixUtils {
 public:
    static void initializeRandomWeights(Eigen::MatrixXd&,
        const bool randomSeed = true);
};

#endif  // MATRIXUTILS_H_
