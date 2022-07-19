#include "activation_function.hpp"

double ReLU::activation_func(double input) {
    return input > 0 ? input : 0;
}

double ReLU::activation_func_deriv(double input) {
    return input > 0 ? 1 : 0;
}
