//
// Created by Samuel Jackson on 29/10/15.
//

#ifndef CYNAPSE_DATASETDESCRIPTOR_H
#define CYNAPSE_DATASETDESCRIPTOR_H

#include <string>
#include <vector>
#include <fstream>

class BatchDatasetDescriptor {
public:
    BatchDatasetDescriptor(const std::string& filename, const int batchSize);

    virtual ~BatchDatasetDescriptor() { m_file.close(); }

    const std::vector<std::string>& getFilenames() const {
        return m_filenames;
    }

    const std::vector<int>& getLabels() const {
        return m_labels;
    }

    const int getBatchSize() const {
        return m_batchSize;
    }

    void next();
    bool hasNext();

private:
    std::ifstream openFileStream(const std::string& filename);
    void skipHeader(const size_t skipLines = 1);
    void resetBatch();
    void readCSVFile();
    void readCSVRow();

private:
    const int m_batchSize;
    std::ifstream m_file;
    std::vector<std::string> m_filenames;
    std::vector<int> m_labels;
};


#endif //CYNAPSE_DATASETDESCRIPTOR_H
