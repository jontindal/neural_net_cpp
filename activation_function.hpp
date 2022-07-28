#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

typedef double (*activation_func_t)(double);
typedef double (*activation_func_deriv_t)(double);

double ReLU(double input);

double deriv_ReLU(double input);

#endif