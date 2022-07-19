#ifndef DATA_READER_HPP
#define DATA_READER_HPP

#include <vector>
#include <fstream>

struct training_data_t {
    unsigned char actual_value;
    std::vector<unsigned char> pixels;
};

class DataReader {
    public:
        DataReader(std::string file_name);

        training_data_t get_data_point();

    private:
        std::ifstream data_file;
};

#endif