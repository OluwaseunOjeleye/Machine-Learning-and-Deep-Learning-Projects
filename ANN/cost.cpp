#include "include/cost.h"

//Quadratic Cost
Matrix<double> quadratic_Cost(const Matrix<double> &true_output, const Matrix<double> &predict_output){
    double cost=0;
	for(int i=0; i<true_output.get_Column(); i++){
        for(int j=0; j<true_output.get_Row(); j++){
        	cost+=pow(predict_output.get_Element(j, i)-true_output.get_Element(j, i), 2);  //Summing up all output nodes
        }
    }
    Matrix<double> Result(1,1, cost*0.5);
    return Result;
}


//derivative of Quadratic Cost
Matrix<double> quadratic_Cost_derivative(const Matrix<double> &true_output, const Matrix<double> &predict_output){
    Matrix<double> Result(true_output.get_Row(), true_output.get_Column());
    for(int i=0; i<true_output.get_Row(); i++){
        for(int j=0; j<true_output.get_Column(); j++){
            Result.set_Element(i, j, predict_output.get_Element(i, j)-true_output.get_Element(i, j));
        }
    }
    return Result;
}
