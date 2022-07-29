#ifndef INITIALIZATION_FUNCTION_HPP
#define INITIALIZATION_FUNCTION_HPP

#include <random>

typedef std::vector<std::vector<double>> (*initialization_func_t)(std::default_random_engine generator, size_t prev_layer_size, size_t next_layer_size);

std::vector<std::vector<double>> he_initialization(std::default_random_engine generator, size_t prev_layer_size, size_t next_layer_size);

#endif