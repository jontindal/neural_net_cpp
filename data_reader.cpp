#include "data_reader.hpp"

#include <sstream>

DataReader::DataReader(std::string file_name) {
    data_file.open(file_name, std::ios::in);

    std::string first_line;
    getline(data_file, first_line);
}

training_data_t DataReader::get_data_point() {
    std::string line;
    getline(data_file, line);

    std::stringstream line_stream(line);
    std::string cell;

    training_data_t result;

    getline(line_stream, cell, ',');
    result.actual_value = std::stoi(cell);

    while (getline(line_stream, cell, ',')) {
        result.pixels.push_back(std::stoi(cell));
    }

    return result;
}