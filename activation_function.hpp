#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

class ReLU {
    public:
        double activation_func(double input);
        double activation_func_deriv(double input);
};

#endif