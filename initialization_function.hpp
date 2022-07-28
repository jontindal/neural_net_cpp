#ifndef INITIALIZATION_FUNCTION_HPP
#define INITIALIZATION_FUNCTION_HPP

#include <random>

class HeInitialization {
    public:
        HeInitialization(double prev_layer_size, double next_layer_size);

        double get_value();

    private:
        std::default_random_engine generator;
        std::normal_distribution<double> distribution;
};

#endif