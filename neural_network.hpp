#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>

#include "initialization_function.hpp"
#include "activation_function.hpp"

constexpr size_t INPUT_SIZE = 784;
constexpr size_t LAYER_ONE_SIZE = 10;
constexpr size_t OUTPUT_SIZE = 10;

struct neural_net_output_t {
    std::vector<std::vector<double>> z_results;
    std::vector<std::vector<double>> a_results;
};

class NeuralNetwork {
    public:
        NeuralNetwork(activation_func_t activation_func, activation_func_deriv_t activation_func_deriv,
                      std::vector<size_t> hidden_layer_sizes, int alpha);

        neural_net_output_t forward_prop(std::vector<unsigned char> input_pixels);

        // double cost_function(unsigned char actual_value, std::vector<double> result);

    private:
        const int number_layers;
        const int alpha;

        const std::vector<size_t> hidden_layer_sizes;

        activation_func_t activation_func;
        activation_func_deriv_t activation_func_deriv;

        std::vector<std::vector<std::vector<double>>> weights;
        std::vector<std::vector<double>> biases;
};

#endif