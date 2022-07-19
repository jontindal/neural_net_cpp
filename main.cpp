#include "data_reader.hpp"
#include "neural_network.hpp"

#include <iostream>

int main() {
    auto data_reader = DataReader("train.csv");
    auto neural_net = NeuralNetwork();

    for (int i=0; i<1; i++) {
        training_data_t data = data_reader.get_data_point();
        std::cout << (int) data.actual_value << "\n";

        std::vector<double> result = neural_net.forward_prop(data.pixels);

        for (auto i : result) {
            std::cout << i << ", ";
        }
        std::cout << "\n";

        std::cout << "Cost = " << neural_net.cost_function(data.actual_value, result) << "\n";
    }
    return 0;
}