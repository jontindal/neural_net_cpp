#include "test_funcs.hpp"

#include "activation_function.hpp"

std::vector<double> get_input_pixels() {
    std::vector<double> input_pixels(INPUT_SIZE);
    return input_pixels;
}

forward_prop_output_t forward_prop(std::vector<std::vector<std::vector<double>>> weights,
                                   std::vector<std::vector<double>> biases,
                                   std::vector<double> input_pixels) {

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

back_prop_output_t back_prop(unsigned char expected_result, forward_prop_output_t forward_prop_output,
                             std::vector<std::vector<std::vector<double>>> weights) {

    std::vector<double> expected_results(OUTPUT_SIZE, 0);
    expected_results[expected_result] = 1;

    back_prop_output_t result;

    std::vector<double> dz1(OUTPUT_SIZE, 0);

    for (size_t i = 0; i < OUTPUT_SIZE; i++) {
        dz1[i] = 2 * (expected_results[i] - forward_prop_output.a_results.back()[i]);
    }

    std::vector<std::vector<double>> dw1 = weights.back();

    for (size_t i = 0; i < dw1.size(); i++) {
        for (size_t j = 0; j < dw1[i].size(); j++) {
            dw1[i][j] = dz1[j] * forward_prop_output.a_results.front()[j];
        }
    }

    std::vector<double> db1 = dz1;

    
    result.dw_results.push_back(dw1);
    result.db_results.push_back(db1);

    return result;
}
