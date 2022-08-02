#include "test_funcs.hpp"

#include <cassert>
#include <random>

#include "activation_function.hpp"

std::vector<double> get_input_pixels() {
    std::vector<double> input_pixels(INPUT_SIZE);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform_distribution(0, 1);

    for (auto& i : input_pixels) {
        i = uniform_distribution(generator);
    }

    return input_pixels;
}

forward_prop_output_t forward_prop(const std::vector<std::vector<std::vector<double>>> weights,
                                   const std::vector<std::vector<double>> biases,
                                   const std::vector<double> input_pixels) {

    forward_prop_output_t result;

    result.z_results.push_back(std::vector<double>());
    result.a_results.push_back(input_pixels);

    for (size_t i = 0; i < biases.size(); i++) {
        std::vector<double> z_result(biases[i].size(), 0);
        
        for (size_t j = 0; j < biases[i].size(); j++) {
            z_result[j] = inner_product(weights[i][j].begin(), weights[i][j].end(), result.a_results.back().begin(), 0);
            z_result[j] += biases[i][j];
        }

        std::vector<double> a_result = (i == biases.size() - 1) ? softmax(z_result) : ReLU(z_result);

        result.z_results.push_back(z_result);
        result.a_results.push_back(a_result);
    }

    return result;
}

static std::vector<double> get_dz_result(const std::vector<std::vector<double>> weights_matrix,
                                         const std::vector<double> z_result,
                                         const std::vector<double> next_layer_dz_result) {

    const size_t current_layer_size = z_result.size();
    const size_t next_layer_size = next_layer_dz_result.size();

    assert (weights_matrix.size() == next_layer_size);
    assert (weights_matrix.front().size() == current_layer_size);

    std::vector<double> dz_result(current_layer_size);

    for (size_t i = 0; i < current_layer_size; i++) {
        for (size_t j = 0; j < next_layer_size; j++) {
            dz_result[i] += weights_matrix[j][i] * next_layer_dz_result[j];
        }
    }

    std::vector<double> activation_func_deriv_result = deriv_ReLU(z_result);

    for (size_t i = 0; i < current_layer_size; i++) {
        dz_result[i] *= activation_func_deriv_result[i];
    }

    return dz_result;
}

static std::vector<std::vector<double>> get_all_dz_results(const size_t no_layers,
                                                           const std::vector<std::vector<std::vector<double>>> weights,
                                                           const std::vector<std::vector<double>> z_results,
                                                           const std::vector<double> expected_results,
                                                           const std::vector<double> actual_results) {
    
    std::vector<std::vector<double>> dz_results(no_layers);

    std::vector<double> final_dz_result(expected_results.size());

    for (size_t i = 0; i < expected_results.size(); i++) {
        final_dz_result[i] = 2 * (expected_results[i] - actual_results[i]);
    }

    dz_results[no_layers - 1] = final_dz_result;

    for (int layer_no = no_layers - 2; layer_no >= 0; layer_no--) {
        dz_results[layer_no] = get_dz_result(weights[layer_no + 1], z_results[layer_no + 1], dz_results[layer_no + 1]);
    }

    return dz_results;
}

back_prop_output_t back_prop(const unsigned char expected_result, const forward_prop_output_t forward_prop_output,
                             const std::vector<std::vector<std::vector<double>>> weights) {

    const size_t no_layers = weights.size();
    assert (forward_prop_output.z_results.size() == no_layers + 1);
    assert (forward_prop_output.a_results.size() == no_layers + 1);

    std::vector<double> expected_results(OUTPUT_SIZE, 0);
    expected_results[expected_result] = 1;

    back_prop_output_t result;

    auto dz_results = get_all_dz_results(no_layers, weights, forward_prop_output.z_results, expected_results, forward_prop_output.a_results.back());

    for (size_t layer_no = 0; layer_no < no_layers; layer_no++) {
        std::vector<std::vector<double>> dw_result(weights[layer_no].size(), std::vector<double>(weights[layer_no].front().size()));

        for (size_t i = 0; i < dw_result.size(); i++) {
            for (size_t j = 0; j < dw_result[i].size(); j++) {
                dw_result[i][j] = dz_results[layer_no][i] * forward_prop_output.a_results[layer_no][j];
            }
        }

        std::vector<double> db_result = dz_results[layer_no];

        result.dw_results.push_back(dw_result);
        result.db_results.push_back(db_result);
    }

    return result;    
}