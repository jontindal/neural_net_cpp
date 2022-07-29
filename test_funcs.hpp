#include <vector>

#include "neural_network.hpp"

std::vector<double> get_input_pixels();

forward_prop_output_t forward_prop(std::vector<std::vector<std::vector<double>>> weights,
                                   std::vector<std::vector<double>> biases,
                                   std::vector<double> input_pixels);
