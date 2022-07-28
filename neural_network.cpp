#include "neural_network.hpp"

#include <cassert>
#include <algorithm>
#include <numeric>

NeuralNetwork::NeuralNetwork(activation_func_t activation_func, activation_func_deriv_t activation_func_deriv,
                             std::vector<size_t> hidden_layer_sizes, double alpha):
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

forward_prop_output_t NeuralNetwork::forward_prop(const std::vector<double>& input_pixels) {
    assert (input_pixels.size() == INPUT_SIZE);

    forward_prop_output_t result;
    result.z_results.push_back(std::vector<double>());
    result.a_results.push_back(input_pixels);

    for (int i = 0; i < number_layers; i++) {
        activation_func_t layer_activation_func = (i == number_layers - 1) ? softmax : activation_func;

        std::vector<double> z_result(INPUT_SIZE, 0);
        for (unsigned int j = 0; j < INPUT_SIZE; j++) {
            assert (weights[i][j].size() == result.a_results.back().size());
            z_result[j] = inner_product(weights[i][j].begin(), weights[i][j].end(), result.a_results.back().begin(), 0);
            z_result[j] += biases[i][j];
        }

        std::vector<double> a_result = layer_activation_func(z_result);

        result.z_results.push_back(z_result);
        result.a_results.push_back(a_result);
    }

    return result;
}

std::vector<double> NeuralNetwork::cost_function_deriv(unsigned char expected_result, const std::vector<double>& actual_results) {
    assert (expected_result < actual_results.size());

    std::vector<double> one_hot_expected_results(actual_results.size(), 0);
    one_hot_expected_results[expected_result] = 1;

    std::vector<double> results(actual_results.size(), 0);

    std::transform(actual_results.begin(), actual_results.end(), one_hot_expected_results.begin(),
                   results.begin(),
                   [](double val_0, double val_1) { return 2 * (val_0 - val_1); });

    return results;
}

back_prop_output_t NeuralNetwork::back_prop(unsigned char expected_result, const std::vector<double>& input_pixels,
                                            const forward_prop_output_t& forward_prop_output) {

    std::vector<std::vector<double>> dz_results;
    dz_results.push_back(cost_function_deriv(expected_result, actual_results));

    for (unsigned int i = 0; i < number_layers - 1; i++) {
        std::vector<double> dz_result(forward_prop_output.dz_results[i].size(), 0);
        dz_results.push_back(dz_result);
    }

    back_prop_output_t result;
    return result;
}