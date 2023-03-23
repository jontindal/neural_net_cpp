#include "activation_function.hpp"

#include <cmath>

std::vector<double> ReLU(std::vector<double> input) {
    std::vector<double> result(input.size(), 0);

    for (unsigned int i = 0; i < input.size(); i++) {
        result[i] = input[i] > 0.0 ? input[i] : 0;
    }
    return result;
}

std::vector<double> deriv_ReLU(std::vector<double> input) {
    std::vector<double> result(input.size(), 0);

    for (unsigned int i = 0; i < input.size(); i++) {
        result[i] = input[i] > 0 ? 1 : 0;
    }
    return result;
}

std::vector<double> softmax(std::vector<double> input) {
    double sum = 0;
    std::vector<double> results(input.size(), 0);
    for (unsigned int i = 0; i < input.size(); i++) {
        double exp_value = exp(input.at(i));
        results.at(i) = exp_value;
        sum += exp_value;
    }

    for (auto& i : results) {
        i = i / sum;
    }

    return results;
}
