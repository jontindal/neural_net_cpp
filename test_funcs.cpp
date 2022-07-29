#include "test_funcs.hpp"

#include "activation_function.hpp"

std::vector<double> get_input_pixels() {
    std::vector<double> input_pixels(INPUT_SIZE);
    return input_pixels;
}

forward_prop_output_t forward_prop(std::vector<std::vector<std::vector<double>>> weights,
                                   std::vector<std::vector<double>> biases,
                                   std::vector<double> input_pixels) {

    std::vector<double> z1(biases[0].size(), 0);

    for (size_t i = 0; i < biases[0].size(); i++) {
        z1[i] = inner_product(weights[0][i].begin(), weights[0][i].end(), input_pixels.begin(), 0);
        z1[i] += biases[0][i];
    }

    std::vector<double> a1 = softmax(z1);

    forward_prop_output_t result;

    result.z_results.push_back(std::vector<double>());
    result.a_results.push_back(input_pixels);
    result.z_results.push_back(z1);
    result.a_results.push_back(a1);

    return result;
}