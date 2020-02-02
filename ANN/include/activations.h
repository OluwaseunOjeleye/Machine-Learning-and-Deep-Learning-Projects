#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_
#include "matrix.h"

//passing activation function as argument to operate on matrix
template <class S> 
Matrix<double> Activation_Function(S function, const Matrix<double> &matrix){
    Matrix<double> Result(matrix.get_Row(), matrix.get_Column());
    for(int i=0; i<matrix.get_Row(); i++){
        for(int j=0; j<matrix.get_Column(); j++){
            Result.set_Element(i, j, function(matrix.get_Element(i, j)));
        }
    }
    return Result;
}

// Activation Functions prototype declaration
double sigmoid(const double param);  
double sigmoid_derivative(const double param);

double relu(const double param);  
double relu_derivative(const double param);

#endif

