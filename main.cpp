#include "data_reader.hpp"

#include <iostream>

int main() {
    auto data_reader = DataReader("train.csv");

    for (int i=0; i<1; i++) {
        training_data_t data = data_reader.get_data_point();
        std::cout << (int) data.actual_value << "\n";

        for (auto j : data.pixels) {
            std::cout << (int) j << ", ";
        }
        std::cout << "\n";
    }
    return 0;
}