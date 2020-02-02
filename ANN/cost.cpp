#include "include/cost.h"

Matrix<double> quadratic_Cost(const Matrix<double> &true_output, const Matrix<double> &predict_output){
	Matrix<double> Result(true_output.get_Column(), 1);
	
	for(int i=0; i<true_output.get_Column(); i++){
		double cost=0;
        for(int j=0; j<true_output.get_Row(); j++){
        	cost+=pow(true_output.get_Element(j, i)-predict_output.get_Element(j, i), 2);
        }
        Result.set_Element(i, 0, cost/2);
    }
    return Result.Transpose();
}

Matrix<double> quadratic_Cost_derivative(const Matrix<double> &true_output, const Matrix<double> &predict_output){
    Matrix<double> Result(true_output.get_Row(), true_output.get_Column());
    for(int i=0; i<true_output.get_Row(); i++){
        for(int j=0; j<true_output.get_Column(); j++){
            Result.set_Element(i, j, true_output.get_Element(j, i)-predict_output.get_Element(j, i));
        }
    }
    return Result;
}
