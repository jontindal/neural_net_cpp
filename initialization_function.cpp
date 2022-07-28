#include "initialization_function.hpp"

HeInitialization::HeInitialization(double prev_layer_size, double next_layer_size):
    distribution(0, sqrt(2 / prev_layer_size)) {}

double const HeInitialization::get_value() {
    return distribution(generator);
}
    