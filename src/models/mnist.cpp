//
// Created by Samuel Jackson on 28/10/2015.
//

#include "mnist.h"

#include <stdio.h>
#include "../core/NeuralNetwork.h"

using namespace cv;

int main(int argc, char** argv )
{
	Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Eigen::VectorXd vec = convert_image_to_vector(image);

	int nodes[3] = { (int) vec.size(), 50, 10 };
	std::vector<int> layout(&nodes[0], &nodes[0]+3);

	NeuralNetwork ann(layout, "sigmoid", false);

	Eigen::MatrixXd example(1, vec.size());
	example << vec.transpose();
	
	Eigen::MatrixXd output(10, 1);
	output << 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;

	ann.train(example, output, 500, 100.0);

	output = ann.feedForward(example);

	std::cout << "========= Final Weights ========" << std::endl;
	std::cout << output << std::endl;
	return 0;
}

Eigen::VectorXd convert_image_to_vector(Mat image)
{
	Eigen::MatrixXd matrix;
	cv2eigen(image, matrix);
	int size = matrix.cols()*matrix.rows();
	Eigen::VectorXd v(Eigen::Map<Eigen::VectorXd>(matrix.data(), size));
	return v;
}


