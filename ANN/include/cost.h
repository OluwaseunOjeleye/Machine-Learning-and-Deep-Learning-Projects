#ifndef COST_H_
#define COST_H_
#include "matrix.h"

//computing cost
template <class R> 
Matrix<double> Cost_Function(R function, const Matrix<double> &true_output, const Matrix<double> &predict_output){    
    return function(true_output, predict_output);
}

Matrix<double> quadratic_Cost(const Matrix<double> &true_output, const Matrix<double> &predict_output);
Matrix<double> quadratic_Cost_derivative(const Matrix<double> &true_output, const Matrix<double> &predict_output);


#endif

