#include "neural_network.hpp"

#include <cmath>
#include <cassert>

std::vector<double> softmax(std::vector<double> inputs) {
    double sum = 0;
    std::vector<double> results(inputs.size(), 0);
    for (unsigned int i = 0; i < inputs.size(); i++) {
        double exp_value = exp(inputs.at(i));
        results.at(i) = exp_value;
        sum += exp_value;
    }

    for (auto& i : results) {
        i = i / sum;
    }

    return results;
}

NeuralNetwork::NeuralNetwork(activation_func_t activation_func, activation_func_deriv_t activation_func_deriv,
                             std::vector<size_t> hidden_layer_sizes, int alpha):
                             number_layers(hidden_layer_sizes.size() + 1),
                             alpha(alpha),
                             hidden_layer_sizes(hidden_layer_sizes) {
        
        this->activation_func = activation_func;
        this->activation_func_deriv = activation_func_deriv;

        std::vector<size_t> total_layers = this->hidden_layer_sizes;
        total_layers.insert(total_layers.begin(), INPUT_SIZE);
        total_layers.insert(total_layers.end(), OUTPUT_SIZE);

        for (unsigned int i = 0; i < total_layers.size() - 1; i++) {
            std::vector<std::vector<double>> weights_matrix(total_layers[i], std::vector<double>(total_layers[i + 1], 0));
            weights.push_back(weights_matrix);

            std::vector<double> biases_array(total_layers[i + 1], 0);
        }
    }

neural_net_output_t NeuralNetwork::forward_prop(std::vector<unsigned char> input_pixels) {
    assert (input_pixels.size() == INPUT_SIZE);

    neural_net_output_t result;

    for (int i = 0; i < number_layers; i++) {
        bool is_last_layer = (i == number_layers - 1);


    }

    return result;
}

// double NeuralNetwork::cost_function(unsigned char actual_value, std::vector<double> result) {
//     assert (result.size() == OUTPUT_SIZE);

//     std::vector<double> desired_output(OUTPUT_SIZE, 0);
//     desired_output.at(actual_value) = 1;

//     double cost_sum = 0;
//     for (int i = 0; i < OUTPUT_SIZE; i++) {
//         cost_sum += pow((desired_output.at(i) - result.at(i)), 2);
//     }

//     return cost_sum;
// }