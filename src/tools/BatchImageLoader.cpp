#include "BatchImageLoader.h"

#include <string>
#include <vector>

namespace fs = boost::filesystem;
using namespace std;

BatchImageLoader::BatchImageLoader(const string& directoryName, const int batchSize, const string& extension)
    : m_batchSize(batchSize), m_extension(extension) {
    m_filenames.reserve(m_batchSize);

    const fs::path directory(directoryName);
    if (fs::exists(directory) && fs::is_directory(directory)) {
        fs::directory_iterator dir_iter(directory);
        m_directoryIter = dir_iter;
    } else {
        throw std::runtime_error(directoryName + " is not a directory");
    }
}

Eigen::MatrixXd BatchImageLoader::next() {
    m_filenames.clear();
    vector<Eigen::VectorXd> images;
    images.reserve(m_batchSize);

    for (int i = 0; i < m_batchSize && m_directoryIter != m_endIter; ++i, ++m_directoryIter) {
        auto path = m_directoryIter->path();

        if (!fs::is_regular_file(m_directoryIter->status())) {
            throw std::runtime_error(path.string() + " is not a valid path.");
        }

        if (path.extension().string() != m_extension) {
            --i;
            continue;
        }

        Eigen::VectorXd img = loadImage(path.string());
        images.push_back(img);
        m_filenames.push_back(path.string());
    }

    Eigen::MatrixXd mat(m_batchSize, 784);
    vector<Eigen::VectorXd>::const_iterator imgIter = images.begin();

    for (int i=0; i < m_batchSize && imgIter != images.end(); ++i, ++imgIter) {
        mat.row(i) = *imgIter;
    }

    return mat;
}

Eigen::VectorXd BatchImageLoader::loadImage(const string& path) {
    cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    return convertImageToVector(image);
}


Eigen::VectorXd BatchImageLoader::convertImageToVector(const cv::Mat& image) {
    Eigen::MatrixXd matrix;
    cv::cv2eigen(image, matrix);
    int size = matrix.cols()*matrix.rows();
    Eigen::VectorXd v(Eigen::Map<Eigen::VectorXd>(matrix.data(), size));
    return v;
}
