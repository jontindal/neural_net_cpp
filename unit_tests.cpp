#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <vector>

#include "initialization_function.hpp"
#include "activation_function.hpp"
#include "neural_network.hpp"

#include "test_funcs.hpp"

TEST_CASE("test_layer_sizes") {
    std::vector<size_t> layers;

    SUBCASE("no_hidden_layers") {layers = {};}
    SUBCASE("one_hidden_layer") {layers =  {13};}
    SUBCASE("two_hidden_layers") {layers =  {13, 12};}
    SUBCASE("three_hidden_layers") {layers =  {13, 12, 11};}

    auto neural_net = NeuralNetwork(layers, he_initialization, ReLU, deriv_ReLU, 0.1);

    std::vector<size_t> total_layers = layers;
    total_layers.insert(total_layers.begin(), INPUT_SIZE);
    total_layers.push_back(OUTPUT_SIZE);

    CHECK (neural_net.get_weights().size() == total_layers.size() - 1);

    for (size_t i = 1; i < neural_net.get_weights().size(); i++) {
        CHECK (neural_net.get_weights()[i - 1].size() == total_layers[i]);

        for (auto& col :  neural_net.get_weights()[i - 1]) {
            CHECK (col.size() == total_layers[i - 1]);
        }
    }

    CHECK (neural_net.get_biases().size() == total_layers.size() - 1);

    for (size_t i = 1; i < neural_net.get_biases().size(); i++) {
        CHECK (neural_net.get_biases()[i - 1].size() == total_layers[i]);
    }
}

TEST_CASE("test_forward_prop") {
    auto input_pixels = get_input_pixels();

    std::vector<size_t> layers;

    SUBCASE("no_hidden_layers") {layers = {};}
    SUBCASE("one_hidden_layer") {layers =  {13};}
    SUBCASE("two_hidden_layers") {layers =  {13, 12};}
    SUBCASE("three_hidden_layers") {layers =  {13, 12, 11};}

    auto neural_net = NeuralNetwork(layers, he_initialization, ReLU, deriv_ReLU, 0.1);

    auto results = neural_net.forward_prop(input_pixels);

    auto other_results = forward_prop(neural_net.get_weights(), neural_net.get_biases(), input_pixels);

    CHECK (results.z_results == other_results.z_results);
    CHECK (results.a_results == other_results.a_results);
}

TEST_CASE("test_back_prop") {
    auto input_pixels = get_input_pixels();

    std::vector<size_t> layers;

    SUBCASE("no_hidden_layers") {layers = {};}
    SUBCASE("one_hidden_layer") {layers =  {13};}
    SUBCASE("two_hidden_layers") {layers =  {13, 12};}
    SUBCASE("three_hidden_layers") {layers =  {13, 12, 11};}

    auto neural_net = NeuralNetwork(layers, he_initialization, ReLU, deriv_ReLU, 0.1);

    forward_prop_output_t forward_prop_output = neural_net.forward_prop(input_pixels);

    back_prop_output_t results = neural_net.back_prop(3, input_pixels, forward_prop_output);
}
   