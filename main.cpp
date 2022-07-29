#include "initialization_function.hpp"
#include "activation_function.hpp"
#include "data_reader.hpp"
#include "neural_network.hpp"

#include <iostream>

int main() {
    auto data_reader = DataReader("train.csv");
    std::vector<size_t> layers {10};
    auto neural_net = NeuralNetwork(layers, he_initialization, ReLU, deriv_ReLU, 0.1);

    return 0;
}