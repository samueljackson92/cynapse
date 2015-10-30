
#ifndef BATCHIMAGELOADER_H_
#define BATCHIMAGELOADER_H_

#include <string>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <string>

#include "BatchDatasetDescriptor.h"


class BatchImageLoader
{
public:
    BatchImageLoader(const std::string& directory, const int batchSize, const int numClasses);
    void next();
    bool hasNext();
    Eigen::MatrixXd loadImages();
    Eigen::MatrixXd loadLabels();

private:
    Eigen::MatrixXd stackVectorsRowwise(const std::vector<Eigen::VectorXd>& vectors);
    Eigen::VectorXd loadImage(const std::string& path);
    Eigen::VectorXd convertImageToVector(const cv::Mat& image);

    BatchDatasetDescriptor m_descriptor;
    const std::string m_directory;
    const int m_numClasses;
};

#endif
