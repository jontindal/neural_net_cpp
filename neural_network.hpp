#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include "activation_function.hpp"

constexpr int INPUT_SIZE = 784;
constexpr int LAYER_ONE_SIZE = 10;
constexpr int OUTPUT_SIZE = 10;

std::vector<double> softmax(std::vector<double> inputs);

struct neural_net_output_t {
    std::vector<double> a1;
    std::vector<double> a2;

    neural_net_output_t():
    a1(LAYER_ONE_SIZE, 0),
    a2(OUTPUT_SIZE, 0)
    {}
};

class NeuralNetwork {
    public:
        NeuralNetwork();

        neural_net_output_t forward_prop(std::vector<unsigned char> input_pixels);

        double cost_function(unsigned char actual_value, std::vector<double> result);

    private:
        ReLU activation_class;

        std::vector<std::vector<double>> w1;
        std::vector<double> b1;

        std::vector<std::vector<double>> w2;
        std::vector<double> b2;
};

#endif