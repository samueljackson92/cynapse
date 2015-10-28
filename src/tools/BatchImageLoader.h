
#ifndef BATCHIMAGELOADER_H_
#define BATCHIMAGELOADER_H_

#include <string>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <string>


class BatchImageLoader
{
    public:
        BatchImageLoader(const std::string& directory, const int batchSize);
        Eigen::MatrixXd next();
        std::vector<std::string> getFilenames() const { return m_filenames; }

    private:
        Eigen::VectorXd loadImage(const std::string& path);
        Eigen::VectorXd convertImageToVector(const cv::Mat& image);

        boost::filesystem::directory_iterator m_directoryIter;
        boost::filesystem::directory_iterator m_endIter;
        std::vector<std::string> m_filenames;
        const int m_batchSize;
};

#endif
