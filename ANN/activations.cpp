#include "include/activations.h"

double sigmoid(const double param){
	return 1/(1+exp(-param));
}

double sigmoid_derivative(const double param){
	return sigmoid(param)*(1-sigmoid(param));
}

double relu(const double param){
	return (param<=0? 0: param);
} 

double relu_derivative(const double param){
	return (param<=0? 0: 1);
}

Matrix<double> Activation_Function(const std::string &string, const Matrix<double> &matrix){
	std::map<std::string, std::function<double(double)>> func_type=
    {
    	{ "sigmoid", sigmoid},
    	{"sigmoid_derivative", sigmoid_derivative},
    	{"relu", relu},
    	{"relu_derivative", relu_derivative}
    };

    Matrix<double> Result(matrix.get_Row(), matrix.get_Column());
    for(int i=0; i<matrix.get_Row(); i++){
        for(int j=0; j<matrix.get_Column(); j++){
            Result.set_Element(i, j, func_type[string](matrix.get_Element(i, j)));
        }
    }
    return Result;
}

//Lambda Function form
/*Matrix<double> Activation_Function(const std::string &string, const Matrix<double> &matrix){
    std::map<std::string, std::function<double(double)>>  func_type =
         {{ "sigmoid", [](double param){return 1/(1+exp(-1*param));}},
          { "sigmoid_derivative", [](double param){return (1/(1+exp(-1*param)))*(1-(1/(1+exp(-1*param))));}},
          { "relu", [](double param){return (param<=0? 0: param);}},
          { "relu_derivative", [](double param){return (param<=0? 0: 1);}}
         };

    Matrix<double> Result(matrix.get_Row(), matrix.get_Column());
    for(int i=0; i<matrix.get_Row(); i++){
        for(int j=0; j<matrix.get_Column(); j++){
            Result.set_Element(i, j, func_type[string](matrix.get_Element(i, j)));
        }
    }
    return Result;
}*/