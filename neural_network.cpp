#include "neural_network.hpp"

#include <cassert>

NeuralNetwork::NeuralNetwork():
    w1(INPUT_SIZE, std::vector<double> (LAYER_ONE_SIZE, 0)),
    b1(LAYER_ONE_SIZE, 0),
    w2(LAYER_ONE_SIZE, std::vector<double> (OUTPUT_SIZE, 0)),
    b2(OUTPUT_SIZE, 0)
    {}

std::vector<double> NeuralNetwork::forward_prop(std::vector<unsigned char> input_pixels) {
    assert (input_pixels.size() == INPUT_SIZE);

    std::vector<double> a1(LAYER_ONE_SIZE, 0);

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < LAYER_ONE_SIZE; j++) {
            a1.at(j) += input_pixels.at(i) * w1.at(i).at(j); 
        }
    }

    for (int i = 0; i < LAYER_ONE_SIZE; i++) {
        a1.at(i) += b1.at(i);
    }

    return a1;
}