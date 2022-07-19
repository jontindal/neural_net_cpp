#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include "activation_function.hpp"

constexpr int INPUT_SIZE = 784;
constexpr int LAYER_ONE_SIZE = 10;
constexpr int OUTPUT_SIZE = 10;

std::vector<double> softmax(std::vector<double> inputs);

class NeuralNetwork {
    public:
        NeuralNetwork();

        std::vector<double> forward_prop(std::vector<unsigned char> input_pixels);

    private:
        ReLU activation_class;

        std::vector<std::vector<double>> w1;
        std::vector<double> b1;

        std::vector<std::vector<double>> w2;
        std::vector<double> b2;
};

#endif