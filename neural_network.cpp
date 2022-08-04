#include "neural_network.hpp"

#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>

#include "math_utils.hpp"

back_prop_output_t::back_prop_output_t(const std::vector<size_t>& hidden_layer_sizes) {
    std::vector<size_t> total_layers = hidden_layer_sizes;
    total_layers.insert(total_layers.begin(), INPUT_SIZE);
    total_layers.insert(total_layers.end(), OUTPUT_SIZE);

    for (unsigned int i = 0; i < total_layers.size() - 1; i++) {
        std::vector<std::vector<double>> dw_matrix(total_layers[i + 1], std::vector<double>(total_layers[i], 0));
        dw_results.push_back(dw_matrix);

        std::vector<double> db_array(total_layers[i + 1], 0);
        db_results.push_back(db_array);
    }
}

void back_prop_output_t::add_new_result(const back_prop_output_t& new_result, double scale_factor) {
    assert (dw_results.size() == new_result.dw_results.size());
    for (size_t layer = 0; layer < dw_results.size(); layer++) {
        assert (dw_results[layer].size() == new_result.dw_results[layer].size());
        for (size_t i = 0; i < dw_results[layer].size(); i++) {
            assert (dw_results[layer][i].size() == new_result.dw_results[layer][i].size());
            for (size_t j = 0; j < dw_results[layer][i].size(); j++) {
                dw_results[layer][i][j] += new_result.dw_results[layer][i][j] * scale_factor;
            }
        }
    }

    assert (db_results.size() == new_result.db_results.size());
    for (size_t layer = 0; layer < db_results.size(); layer++) {
        assert (db_results[layer].size() == new_result.db_results[layer].size());
        for (size_t i = 0; i < db_results[layer].size(); i++) {
            db_results[layer][i] += new_result.db_results[layer][i] * scale_factor;
        }
    }
}

NeuralNetwork::NeuralNetwork(std::vector<size_t> hidden_layer_sizes, initialization_func_t initialization_func,
                             activation_func_t activation_func, activation_func_deriv_t activation_func_deriv,
                             double alpha):
                             number_layers(hidden_layer_sizes.size() + 1),
                             alpha(alpha),
                             hidden_layer_sizes(hidden_layer_sizes) {
        
    this->initialization_func = initialization_func;
    this->activation_func = activation_func;
    this->activation_func_deriv = activation_func_deriv;

    std::vector<size_t> total_layers = this->hidden_layer_sizes;
    total_layers.insert(total_layers.begin(), INPUT_SIZE);
    total_layers.insert(total_layers.end(), OUTPUT_SIZE);

    std::default_random_engine generator;

    for (unsigned int i = 0; i < total_layers.size() - 1; i++) {
        std::vector<std::vector<double>> weights_matrix = he_initialization(generator, total_layers[i], total_layers[i + 1]);
        weights.push_back(weights_matrix);

        std::vector<double> biases_array(total_layers[i + 1], 0);
        biases.push_back(biases_array);
    }
}

const std::vector<std::vector<std::vector<double>>> NeuralNetwork::get_weights() {
    return weights;
}

const std::vector<std::vector<double>> NeuralNetwork::get_biases() {
    return biases;
}

forward_prop_output_t NeuralNetwork::forward_prop(const std::vector<double>& input_pixels) {
    assert (input_pixels.size() == INPUT_SIZE);

    forward_prop_output_t result;
    result.z_results.push_back(std::vector<double>());
    result.a_results.push_back(input_pixels);

    for (int i = 0; i < number_layers; i++) {
        activation_func_t layer_activation_func = (i == number_layers - 1) ? softmax : activation_func;

        std::vector<double> z_result = dot_product<false>(weights[i], result.a_results.back());
        for (unsigned int j = 0; j < biases[i].size(); j++) {
            assert (weights[i][j].size() == result.a_results.back().size());
            z_result[j] += biases[i][j];
        }

        std::vector<double> a_result = layer_activation_func(z_result);

        result.z_results.push_back(z_result);
        result.a_results.push_back(a_result);
    }

    return result;
}

unsigned char NeuralNetwork::get_best_guess(const std::vector<double>& forward_prop_final_result) {
    return std::max_element(forward_prop_final_result.begin(), forward_prop_final_result.end()) - forward_prop_final_result.begin();
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

    std::vector<std::vector<double>> dz_results(number_layers);
    dz_results[number_layers - 1] = cost_function_deriv(expected_result, forward_prop_output.a_results.back());

    for (int i = number_layers - 2; i > -1; i--) {
        std::vector<double> dz_result = dot_product<true>(weights[i + 1], dz_results[i + 1]);

        std::vector<double> temp = activation_func_deriv(forward_prop_output.z_results[i + 1]);

        std::transform(dz_result.begin(), dz_result.end(), temp.begin(), dz_result.begin(),
                       [](double val_0, double val_1) { return val_0 * val_1; });

        dz_results[i] = dz_result;
    }

    back_prop_output_t result(hidden_layer_sizes);

    for (int i = 0; i < number_layers; i++) {
        std::vector<std::vector<double>> dw_result(weights[i].size(), std::vector<double> (weights[i][0].size(), 0));

        for (size_t j = 0; j < weights[i].size(); j++) {
            for (size_t k = 0; k < weights[i][j].size(); k++) {
                dw_result[j][k] = dz_results[i][j] * forward_prop_output.a_results[i][k];
            }
        }
        result.dw_results[i] = dw_result;
        result.db_results[i] = dz_results[i];
    }

    return result;
}

void NeuralNetwork::update_params(const back_prop_output_t& back_prop_output) {
    for (int layer = 0; layer < number_layers; layer++) {
        for (size_t i = 0; i < weights[layer].size(); i++) {
            for (size_t j = 0; j < weights[layer][i].size(); j++) {
                weights[layer][i][j] -= alpha * back_prop_output.dw_results[layer][i][j];
            }
        }

        for (size_t i = 0; i < biases[layer].size(); i++) {
            biases[layer][i] -= alpha * back_prop_output.db_results[layer][i];
        }
    }
}

std::vector<double> NeuralNetwork::gradient_descent(const std::vector<training_data_t> training_data,
                                                    unsigned long iterations,
                                                    bool print_progress) {
    std::vector<double> accuracies(iterations);

    for (size_t i = 0; i < iterations; i++) {
        size_t correct_guesses = 0;
        back_prop_output_t avg_output(hidden_layer_sizes);

        for (auto& data_point : training_data) {
            auto double_vector = to_double_vector(data_point.pixels);
            forward_prop_output_t forward_prop_output = forward_prop(double_vector);
            back_prop_output_t back_prop_output = back_prop(data_point.actual_value, double_vector, forward_prop_output);

            avg_output.add_new_result(back_prop_output, (1. / training_data.size()));

            correct_guesses += (get_best_guess(forward_prop_output.a_results.back()) == data_point.actual_value) ? 1 : 0;
        }

        update_params(avg_output);
        accuracies[i] = (double) correct_guesses / training_data.size();

        if ((i % 20 == 0) && print_progress) {
            std::cout << "Accuracy on iteration " << i << ": " << accuracies[i] << "\n";
        }
    }

    return accuracies;
}