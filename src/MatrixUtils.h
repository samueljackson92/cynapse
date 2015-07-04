
#include <Eigen/Dense>
#include <vector>

class MatrixUtils {

 public:
    static void initializeRandomWeights(MatrixXd_ptr);
    static void applyFunction(MatrixXd_ptr, std::function<double(double)>);
};
