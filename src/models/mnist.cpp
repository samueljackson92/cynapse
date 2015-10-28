//
// Created by Samuel Jackson on 28/10/2015.
//

#include "mnist.h"

#include <stdio.h>

using namespace cv;

int main(int argc, char** argv )
{
	Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Eigen::VectorXf vec = convert_image_to_vector(image);
	std::cout << vec.size() << std::endl;
	return 0;
}

Eigen::VectorXf convert_image_to_vector(Mat image)
{
	Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> matrix;
	cv2eigen(image, matrix);
	int size = matrix.cols()*matrix.rows();
	Eigen::VectorXf v(Eigen::Map<Eigen::VectorXf>(matrix.data(), size));
	return v;
}


