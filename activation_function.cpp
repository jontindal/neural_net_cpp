#include "activation_function.hpp"

double ReLU(double input) {
    return input > 0 ? input : 0;
}

double deriv_ReLU(double input) {
    return input > 0 ? 1 : 0;
}
