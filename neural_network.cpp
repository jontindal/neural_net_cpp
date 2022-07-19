#include "neural_network.hpp"

#include <cmath>
#include <cassert>

std::vector<double> softmax(std::vector<double> inputs) {
    double sum = 0;
    std::vector<double> results(inputs.size(), 0);
    for (unsigned int i = 0; i < inputs.size(); i++) {
        double exp_value = exp(inputs.at(i));
        results.at(i) = exp_value;
        sum += exp_value;
    }

    for (auto& i : results) {
        i = i / sum;
    }

    return results;
}

NeuralNetwork::NeuralNetwork():
    w1(INPUT_SIZE, std::vector<double> (LAYER_ONE_SIZE, 0)),
    b1(LAYER_ONE_SIZE, 0),
    w2(LAYER_ONE_SIZE, std::vector<double> (OUTPUT_SIZE, 0)),
    b2(OUTPUT_SIZE, 0)
    {}

std::vector<double> NeuralNetwork::forward_prop(std::vector<unsigned char> input_pixels) {
    assert (input_pixels.size() == INPUT_SIZE);

    std::vector<double> z1(LAYER_ONE_SIZE, 0);

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < LAYER_ONE_SIZE; j++) {
            z1.at(j) += input_pixels.at(i) * w1.at(i).at(j); 
        }
    }

    for (int i = 0; i < LAYER_ONE_SIZE; i++) {
        z1.at(i) += b1.at(i);
    }

    std::vector<double> a1(LAYER_ONE_SIZE, 0);

    for (int i = 0; i < LAYER_ONE_SIZE; i++) {
        a1.at(i) = activation_class.activation_func(z1.at(i));
    }

    std::vector<double> z2(OUTPUT_SIZE, 0);

    for (int i = 0; i < LAYER_ONE_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            z2.at(j) += a1.at(i) * w2.at(i).at(j);
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        z2.at(i) += b2.at(i);
    }

    std::vector<double> a2 = softmax(z2);

    return a2;
}

double NeuralNetwork::cost_function(unsigned char actual_value, std::vector<double> result) {
    assert (result.size() == OUTPUT_SIZE);

    std::vector<double> desired_output(OUTPUT_SIZE, 0);
    desired_output.at(actual_value) = 1;

    double cost_sum = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        cost_sum += pow((desired_output.at(i) - result.at(i)), 2);
    }

    return cost_sum;
}