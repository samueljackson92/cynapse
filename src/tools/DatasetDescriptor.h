//
// Created by Samuel Jackson on 29/10/15.
//

#ifndef CYNAPSE_DATASETDESCRIPTOR_H
#define CYNAPSE_DATASETDESCRIPTOR_H

#include <string>
#include <vector>
#include <fstream>

class DatasetDescriptor {
public:
    DatasetDescriptor(const std::string& filename, const int batchSize);

    virtual ~DatasetDescriptor() { }

    const std::vector<std::string>& getFilenames() const {
        return m_filenames;
    }

    const std::vector<int>& getLabels() const {
        return m_labels;
    }

    void next();

private:
    std::ifstream openFileStream(const std::string& filename);
    void skipHeader(const size_t skipLines = 1);
    void resetBatch();
    void readCSVFile();
    void readCSVRow();

    const int m_batchSize;
    std::ifstream m_file;
    std::vector<std::string> m_filenames;
    std::vector<int> m_labels;
};


#endif //CYNAPSE_DATASETDESCRIPTOR_H
