#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_
#include "matrix.h"
#include <functional>
#include <map>

// Activation Functions prototype declaration
double sigmoid(const double param);  
double sigmoid_derivative(const double param);

double relu(const double param);  
double relu_derivative(const double param);

Matrix<double> Activation_Function(const std::string &string, const Matrix<double> &matrix);


#endif