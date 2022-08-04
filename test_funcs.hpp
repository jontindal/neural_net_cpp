#include <vector>

#include "neural_network.hpp"

std::vector<double> get_input_pixels();

forward_prop_output_t forward_prop(const std::vector<std::vector<std::vector<double>>> weights,
                                   const std::vector<std::vector<double>> biases,
                                   const std::vector<double> input_pixels);

back_prop_output_t back_prop(const unsigned char expected_result, const forward_prop_output_t forward_prop_output,
                             const std::vector<size_t> hidden_layer_sizes,
                             const std::vector<std::vector<std::vector<double>>> weights);
