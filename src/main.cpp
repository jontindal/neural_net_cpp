#include "initialization_function.hpp"
#include "activation_function.hpp"
#include "data_reader.hpp"
#include "neural_network.hpp"

#include <iostream>

int main() {
    auto data_reader = DataReader("train.csv");
    std::vector<size_t> layers {10};
    auto neural_net = NeuralNetwork(layers, he_initialization, ReLU, deriv_ReLU, 0.1);

    std::vector<training_data_t> training_data;
    for (int i = 0; i < 42000; i++) {
        training_data.push_back(data_reader.get_data_point());
    }
    std::cout << "Starting\n";

    neural_net.gradient_descent(training_data, 500);

    return 0;
}