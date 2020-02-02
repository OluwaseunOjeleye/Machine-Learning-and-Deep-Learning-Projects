#include "include/activations.h"

double sigmoid(const double param){
	return 1/(1+exp(-1*param));
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