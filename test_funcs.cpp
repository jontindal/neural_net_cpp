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