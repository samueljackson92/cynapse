#include "BatchImageLoader.h"

#include <string>
#include <vector>

namespace fs = boost::filesystem;
using namespace std;
using namespace Eigen;

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

MatrixXd BatchImageLoader::next() {
    nextFilenameBatch();
    auto images = loadImages(m_filenames);
    return stackVectorsRowwise(images);
}

void BatchImageLoader::nextFilenameBatch() {
    m_filenames.clear();

    for (int i = 0; i < m_batchSize && m_directoryIter != m_endIter; ++i, ++m_directoryIter) {
        auto path = m_directoryIter->path();

        if (!fs::is_regular_file(m_directoryIter->status())) {
            throw std::runtime_error(path.string() + " is not a valid path.");
        }

        if (path.extension().string() != m_extension) {
            --i;
            continue;
        }

        m_filenames.push_back(path.string());
    }
}

vector<VectorXd> BatchImageLoader::loadImages(const vector<string>& filenames) {
    vector<VectorXd> images;
    images.reserve(m_batchSize);

    for(auto fileIter = m_filenames.cbegin(); fileIter != m_filenames.cend(); ++fileIter) {
        VectorXd img = loadImage(*fileIter);
        images.push_back(img);
    }

    return images;
}

MatrixXd BatchImageLoader::stackVectorsRowwise(const std::vector<VectorXd>& vectors) {
    vector<VectorXd>::const_iterator imgIter = vectors.begin();
    if (imgIter != vectors.end()) {
        MatrixXd mat(m_batchSize, vectors[0].size());
        for (int i = 0; i < m_batchSize && imgIter != vectors.end(); ++i, ++imgIter) {
            mat.row(i) = *imgIter;
        }
        return mat;
    } else {
        return MatrixXd::Zero(m_batchSize, 0);
    }
}

VectorXd BatchImageLoader::loadImage(const string& path) {
    cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    return convertImageToVector(image);
}


VectorXd BatchImageLoader::convertImageToVector(const cv::Mat& image) {
    MatrixXd matrix;
    cv::cv2eigen(image, matrix);
    int size = matrix.cols()*matrix.rows();
    VectorXd v(Map<VectorXd>(matrix.data(), size));
    return v;
}
