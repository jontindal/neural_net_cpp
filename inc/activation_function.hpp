#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <vector>

typedef std::vector<double> (*activation_func_t)(std::vector<double>);
typedef std::vector<double> (*activation_func_deriv_t)(std::vector<double>);

std::vector<double> ReLU(std::vector<double> input);

std::vector<double> deriv_ReLU(std::vector<double> input);

std::vector<double> softmax(std::vector<double> input);

#endif