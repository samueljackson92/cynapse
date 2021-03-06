//
// Created by Samuel Jackson on 29/10/15.
//

#include "BatchDatasetDescriptor.h"

#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <iostream>

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace boost;

BatchDatasetDescriptor::BatchDatasetDescriptor(const string &filename, const int batchSize)
        : m_batchSize(batchSize) {

    m_file = openFileStream(filename);
    skipHeader();
}

void BatchDatasetDescriptor::next() {
    resetBatch();
    readCSVFile();
}

void BatchDatasetDescriptor::resetBatch() {
    m_filenames.clear();
    m_filenames.reserve(m_batchSize);

    m_labels.clear();
    m_labels.reserve(m_batchSize);
}

void BatchDatasetDescriptor::readCSVFile() {
    for(int i=0; i < m_batchSize; ++i) {
        readCSVRow();
    }
}

void BatchDatasetDescriptor::readCSVRow() {
    typedef tokenizer< escaped_list_separator<char> > Tokenizer;
    vector<string> vec;
    string line;

    if(!getline(m_file,line)) {
        return;
    }

    Tokenizer tok(line);
    vec.assign(tok.begin(),tok.end());

    if (vec.size() < 3) {
        return;
    }

    m_labels.push_back(lexical_cast<int>(vec[1]));
    m_filenames.push_back(vec[2]);
}

void BatchDatasetDescriptor::skipHeader(const size_t skipLines) {
    for (size_t i = 0; i < skipLines; ++i) {
        m_file.ignore(numeric_limits<streamsize>::max(), '\n');
    }
}

std::ifstream BatchDatasetDescriptor::openFileStream(const std::string &filename) {
    ifstream in(filename.c_str());

    if (!in.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }
    return in;
}

bool BatchDatasetDescriptor::hasNext() {
    return (m_file && m_file.peek() != EOF);
}
