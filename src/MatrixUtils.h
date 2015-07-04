
#include <Eigen/Dense>
#include <vector>

class MatrixUtils {

 public:
    static void initializeRandomWeights(MatrixXd_ptr,
        const bool randomSeed = true);
    static void applyFunction(MatrixXd_ptr, std::function<double(double)>);
};
