#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_
#include "activations.h"
#include "cost.h"
#include <string.h>

class NeuralNetwork{
	public:
		NeuralNetwork(std::vector<int> no_of_layers);//excluding input layers
		~NeuralNetwork();

		void Train(Matrix<double> X_train, Matrix<double> Y_train, double alpha, std::string h_Layer_Activation, std::string output_Layer_Activation);
		void Predict(Matrix<double> X_test);

	private:
		void initialize_Parameters();
		void forward_Propagation();
		void compute_Cost();
		void backward_Propagation();
		void update_Parameters();

		std::vector<Matrix<double>> hidden_layers; //including X as h0
		std::vector<Matrix<double>> Weights; 
		std::vector<Matrix<double>> Biases;
		std::vector<int> hidden_layers_vector;
		Matrix<double> true_Output;
		Matrix<double> cost;
		int no_of_hidden_layers;
		int no_of_samples;
		int no_of_features;
};

NeuralNetwork::NeuralNetwork(std::vector<int> no_of_layers){
	this->no_of_samples=0;
	this->no_of_features=0;
	this->hidden_layers_vector=no_of_layers;
	this->no_of_hidden_layers=this->hidden_layers_vector.size();
}

NeuralNetwork::~NeuralNetwork(){

}

void NeuralNetwork::Train(Matrix<double> X_train, Matrix<double> Y_train, double alpha, 
							std::string h_Layer_Activation, std::string output_Layer_Activation){

	this->no_of_samples=X_train.get_Row();
	this->no_of_features=X_train.get_Column();

	//Initiaizing Hidden Layer Matrices
	this->hidden_layers.resize(this->no_of_hidden_layers+1);
	this->hidden_layers[0]=X_train;	//X=h0
	for(int i=1; i<this->no_of_hidden_layers+1; i++){
		this->hidden_layers[i].resize(this->hidden_layers_vector[i-1], this->no_of_samples);	//h1 to output(hn)
	}

	initialize_Parameters();
	forward_Propagation();
	this->true_Output=Y_train;
	this->true_Output.print();
	compute_Cost();

}


void NeuralNetwork::initialize_Parameters(){
	//initializing Weight
	this->Weights.resize(this->no_of_hidden_layers);
	for(int i=0; i<this->no_of_hidden_layers; i++){
		i==0? this->Weights[i].resize(this->hidden_layers_vector[i], this->no_of_features):	//W1
			this->Weights[i].resize(this->hidden_layers_vector[i], this->hidden_layers_vector[i-1]); 	//W2-Wn
		this->Weights[i].generate_Random_Elements();
	}

	//initializing Bias
	this->Biases.resize(this->no_of_hidden_layers);
	for(int i=0; i<this->no_of_hidden_layers; i++){
		this->Biases[i].resize(this->hidden_layers_vector[i], this->no_of_samples);	//B1-Bn
		this->Biases[i].generate_Random_Elements();
	}
}

void NeuralNetwork::forward_Propagation(){
	this->hidden_layers[0].print();
	for(int i=1; i<this->no_of_hidden_layers+1; i++){
		i!=this->no_of_hidden_layers? this->hidden_layers[i]=Activation_Function(relu, (this->Weights[i-1]*this->hidden_layers[i-1])+this->Biases[i-1]):
			this->hidden_layers[i]=Activation_Function(sigmoid, (this->Weights[i-1]*this->hidden_layers[i-1])+this->Biases[i-1]);
		this->hidden_layers[i].print();
	}
}

void NeuralNetwork::compute_Cost(){	//Quadratic Cost
	this->cost=Cost_Function(quadratic_Cost, this->true_Output, Activation_Function(sigmoid, this->hidden_layers[no_of_hidden_layers]));
	this->cost.print();
}

void NeuralNetwork::backward_Propagation(){

}

void NeuralNetwork::update_Parameters(){

}

#endif