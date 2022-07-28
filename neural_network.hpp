#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>

#include "initialization_function.hpp"
#include "activation_function.hpp"

constexpr size_t INPUT_SIZE = 784;
constexpr size_t LAYER_ONE_SIZE = 10;
constexpr size_t OUTPUT_SIZE = 10;

struct forward_prop_output_t {
    std::vector<std::vector<double>> z_results;
    std::vector<std::vector<double>> a_results;
};

struct back_prop_output_t {
    std::vector<std::vector<std::vector<double>>> dw_results;
    std::vector<std::vector<double>> db_results;
}

class NeuralNetwork {
    public:
        NeuralNetwork(activation_func_t activation_func, activation_func_deriv_t activation_func_deriv,
                      std::vector<size_t> hidden_layer_sizes, double alpha);

        forward_prop_output_t forward_prop(const std::vector<double>& input_pixels);

        std::vector<double> cost_function_deriv(unsigned char expected_result, const std::vector<double>& actual_results);

        back_prop_output_t back_prop(unsigned char expected_result, const std::vector<double>& input_pixels,
                                     const forward_prop_output_t& forward_prop_output);

    private:
        const int number_layers;
        const double alpha;

        const std::vector<size_t> hidden_layer_sizes;

        activation_func_t activation_func;
        activation_func_deriv_t activation_func_deriv;

        std::vector<std::vector<std::vector<double>>> weights;
        std::vector<std::vector<double>> biases;
};

#endif