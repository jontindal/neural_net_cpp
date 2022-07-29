#include <iostream>
#include <vector>

#include "initialization_function.hpp"
#include "activation_function.hpp"
#include "neural_network.hpp"

#include "test_funcs.hpp"

bool test_layer_sizes() {
    bool test_failed = false;

    std::vector<size_t> layers;

    for (int i = 13; i > 10; i--) {
        layers.push_back(i);
        
        auto neural_net = NeuralNetwork(layers, he_initialization, ReLU, deriv_ReLU, 0.1);

        std::vector<size_t> total_layers = layers;
        total_layers.insert(total_layers.begin(), INPUT_SIZE);
        total_layers.push_back(OUTPUT_SIZE);

        test_failed |= (neural_net.get_weights().size() != total_layers.size() - 1);

        for (size_t i = 1; i < neural_net.get_weights().size(); i++) {
            test_failed |= (neural_net.get_weights()[i - 1].size() != total_layers[i]);

            for (auto& col :  neural_net.get_weights()[i - 1]) {
                test_failed |= (col.size() != total_layers[i - 1]);
            }
        }

        test_failed |= (neural_net.get_biases().size() != total_layers.size() - 1);

        for (size_t i = 1; i < neural_net.get_biases().size(); i++) {
            test_failed |= (neural_net.get_biases()[i - 1].size() != total_layers[i]);
        }
    }

    return test_failed;
}

bool test_forward_prop() {
    bool test_failed = false;

    auto input_pixels = get_input_pixels();

    std::vector<size_t> layers;

    for (int i = 13; i > 10; i--) {
        layers.push_back(i);

        auto neural_net = NeuralNetwork(layers, he_initialization, ReLU, deriv_ReLU, 0.1);

        auto results = neural_net.forward_prop(input_pixels);

        auto other_results = forward_prop(neural_net.get_weights(), neural_net.get_biases(), input_pixels);

        test_failed |= (results.z_results != other_results.z_results);
        test_failed |= (results.a_results != other_results.a_results);
    }

    return test_failed;
}

int main() {
    std::cout << test_layer_sizes() << '\n';
    std::cout << test_forward_prop() << '\n';

    return 0;
}