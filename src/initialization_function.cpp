#include "initialization_function.hpp"

std::vector<std::vector<double>> he_initialization(std::default_random_engine generator, size_t prev_layer_size, size_t next_layer_size) {
    std::vector<std::vector<double>> result(next_layer_size, std::vector<double>(prev_layer_size, 0));

    std::normal_distribution<double> distribution(0.0, sqrt(static_cast<double>(2) / prev_layer_size));

    for (auto& row : result) {
        for (auto& value : row) {
            value = distribution(generator);
        }
    }

    return result;
}
    