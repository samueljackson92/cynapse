#include "BatchImageLoader.h"

#include <string>
#include <vector>

namespace fs = boost::filesystem;
using namespace std;
using namespace Eigen;

const size_t MAX_PIXEL_VALUE = 255;
const string DATASET_CSV_NAME  = "/dataset.csv";

BatchImageLoader::BatchImageLoader(const string& directoryName, const int batchSize, const int numClasses)
    : m_numClasses(numClasses), m_directory(directoryName),
      m_descriptor(directoryName + DATASET_CSV_NAME, batchSize) {

    const fs::path directory(directoryName);
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        throw std::runtime_error(directoryName + " is not a directory");
    }
}

void BatchImageLoader::next() {
    m_descriptor.next();
}

MatrixXd BatchImageLoader::loadImages() {
    const int batchSize = m_descriptor.getBatchSize();
    auto filenames = m_descriptor.getFilenames();

    vector<VectorXd> images;
    images.reserve(batchSize);

    for(auto fileIter = filenames.cbegin(); fileIter != filenames.cend(); ++fileIter) {
        fs::path full_path = m_directory / fs::path(*fileIter);
        VectorXd img = loadImage(full_path.string());
        images.push_back(img);
    }

    return stackVectorsRowwise(images);
}

Eigen::MatrixXd BatchImageLoader::loadLabels() {
    const int batchSize = m_descriptor.getBatchSize();
    auto labels = m_descriptor.getLabels();

    Eigen::MatrixXd mat(m_numClasses, labels.size());

    size_t i = 0;
    for (auto it = labels.cbegin(); it != labels.cend(); ++it, ++i) {
        int label = *it;
        Eigen::VectorXd v = Eigen::VectorXd::Zero(mat.rows());
        v(label) = 1;
        mat.col(i) = v;
    }

    return mat;
}

MatrixXd BatchImageLoader::stackVectorsRowwise(const std::vector<VectorXd>& vectors) {
    int batchSize = m_descriptor.getBatchSize();
    vector<VectorXd>::const_iterator imgIter = vectors.begin();

    if (imgIter != vectors.end()) {
        MatrixXd mat(batchSize, vectors[0].size());
        for (int i = 0; i < batchSize && imgIter != vectors.end(); ++i, ++imgIter) {
            mat.row(i) = *imgIter;
        }

        //Normalize image to unity
        mat = mat / 255;

        //Append "bias" row to input
        mat.conservativeResize(mat.rows(), mat.cols()+1);
        mat.col(mat.cols()-1) = Eigen::VectorXd::Ones(mat.rows());

        return mat;
    } else {
        return MatrixXd::Zero(batchSize, 0);
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

bool BatchImageLoader::hasNext() {
    return m_descriptor.hasNext();
}
