#include "neural_network.hpp"

#include <cassert>
#include <numeric>

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
            std::vector<std::vector<double>> weights_matrix(total_layers[i + 1], std::vector<double>(total_layers[i], 0));
            weights.push_back(weights_matrix);

            std::vector<double> biases_array(total_layers[i + 1], 0);
            biases.push_back(biases_array);
        }
    }

neural_net_output_t NeuralNetwork::forward_prop(const std::vector<double> input_pixels) {
    assert (input_pixels.size() == INPUT_SIZE);

    neural_net_output_t result;
    result.z_results.push_back(std::vector<double>());
    result.a_results.push_back(input_pixels);

    for (int i = 0; i < number_layers; i++) {
        activation_func_t layer_activation_func = (i == number_layers - 1) ? softmax : activation_func;

        std::vector<double> z_result(INPUT_SIZE, 0);
        for (unsigned int j = 0; j < INPUT_SIZE; j++) {
            assert (weights[i][j].size() == result.a_results.back().size());
            z_result[j] = inner_product(weights[i][j].begin(), weights[i][j].end(), result.a_results.back().begin(), 0);
        }

        std::vector<double> a_result = layer_activation_func(z_result);

        result.z_results.push_back(z_result);
        result.a_results.push_back(a_result);
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