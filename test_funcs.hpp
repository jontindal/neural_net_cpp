#include <vector>

#include "neural_network.hpp"

std::vector<double> get_input_pixels();

forward_prop_output_t forward_prop(std::vector<std::vector<std::vector<double>>> weights,
                                   std::vector<std::vector<double>> biases,
                                   std::vector<double> input_pixels);

back_prop_output_t back_prop(unsigned char expected_result, forward_prop_output_t forward_prop_output,
                             std::vector<std::vector<std::vector<double>>> weights);
