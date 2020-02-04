#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_
#include "activations.h"
#include "cost.h"
#include <string.h>


//Neural Network with Stochastic Gradient Descent as Optimizer
class NeuralNetwork{
	public:
		NeuralNetwork(std::vector<int> no_of_layers);//excluding input layers
		~NeuralNetwork();

		void Train(Matrix<double> X_train, Matrix<double> Y_train, double alpha, std::string h_Layer_Activation, std::string output_Layer_Activation);
		Matrix<double> Predict(Matrix<double> X_test);

	private:
		void initialize_Parameters();
		void forward_Propagation();
		void compute_Cost();
		void backward_Propagation();

		//For Forward Propagation
		std::vector<Matrix<double>> hidden_layers; //Array of Matrix for storing hidden layers including X as h0 i.e H0 to Hn
		std::vector<Matrix<double>> Weights; //Array of Matrix for weights between each layers i.e W0 to Wn-1
		std::vector<Matrix<double>> Biases; //Array of Matrix for Biases	i.e B0 to Bn-1

		//For backpropagation
		std::vector<Matrix<double>> layers_error; //Array of Matrix for errors at each layers
		std::vector<Matrix<double>> Weight_gradient; //Array of Matrix for gradient of weights
		std::vector<Matrix<double>> Bias_gradient; //Array of Matrix for gradients of Biases

		double alpha;

		std::vector<int> hidden_layers_vector;
		Matrix<double> true_Output;
		Matrix<double> cost;
		int no_of_hidden_layers;
		int no_of_samples;
		int no_of_features;

		bool is_trained;
};

NeuralNetwork::NeuralNetwork(std::vector<int> no_of_layers){
	this->no_of_samples=1;	//Stochastic Gradient Descent is used i.e batch size=1
	this->no_of_features=0;
	this->hidden_layers_vector=no_of_layers;
	this->no_of_hidden_layers=this->hidden_layers_vector.size();
	this->alpha=0.01;
	this->is_trained=false;
}

NeuralNetwork::~NeuralNetwork(){

}

void NeuralNetwork::Train(Matrix<double> X_train, Matrix<double> Y_train, double alpha, 
							std::string h_Layer_Activation, std::string output_Layer_Activation){

	this->alpha=alpha;
	this->no_of_features=X_train.get_Column();

	//Initiaizing Hidden Layer Matrices
	this->hidden_layers.resize(this->no_of_hidden_layers+1);
	this->hidden_layers[0]=X_train.Transpose();	//h0=transpose(X) and its dimension is no_of_feature X no_of_sample
	for(int i=1; i<this->no_of_hidden_layers+1; i++){
		this->hidden_layers[i].resize(this->hidden_layers_vector[i-1], this->no_of_samples);	//h1 to output(hn)
	}

	if(!this->is_trained)initialize_Parameters();
	forward_Propagation();
	this->true_Output=Y_train.Transpose();
	//this->true_Output.print();
	//compute_Cost();
	backward_Propagation();
}


void NeuralNetwork::initialize_Parameters(){
	//initializing Weight
	this->Weights.resize(this->no_of_hidden_layers);
	for(int i=0; i<this->no_of_hidden_layers; i++){
		i==0? this->Weights[i].resize(this->hidden_layers_vector[i], this->no_of_features):	//W1
			this->Weights[i].resize(this->hidden_layers_vector[i], this->hidden_layers_vector[i-1]); 	//W2-Wn
		this->Weights[i].generate_Random_Elements();

		//this->Weights[i].print();
	}

	//initializing Bias
	this->Biases.resize(this->no_of_hidden_layers);
	for(int i=0; i<this->no_of_hidden_layers; i++){
		this->Biases[i].resize(this->hidden_layers_vector[i], this->no_of_samples);	//B1-Bn
		this->Biases[i].generate_Random_Elements();
	}
	this->is_trained=true;
}

void NeuralNetwork::forward_Propagation(){
	//this->hidden_layers[0].print();
	for(int i=1; i<this->no_of_hidden_layers+1; i++){
		i!=this->no_of_hidden_layers? this->hidden_layers[i]=Activation_Function(relu, (this->Weights[i-1]*this->hidden_layers[i-1])+this->Biases[i-1]):
			this->hidden_layers[i]=Activation_Function(sigmoid, (this->Weights[i-1]*this->hidden_layers[i-1])+this->Biases[i-1]);
		//this->hidden_layers[i].print();
	}
}

void NeuralNetwork::compute_Cost(){	//Quadratic Cost
	this->cost=Cost_Function(quadratic_Cost, this->true_Output, this->hidden_layers[no_of_hidden_layers]);
	//this->cost.print();
}

void NeuralNetwork::backward_Propagation(){
	//initialization
	this->layers_error.resize(this->no_of_hidden_layers);
	this->Weight_gradient.resize(this->no_of_hidden_layers);
	this->Bias_gradient.resize(this->no_of_hidden_layers);

	//computing errors at each layers
	for (int i=this->no_of_hidden_layers-1; i>=0; i--){
		std::cout<<"("<<quadratic_Cost_derivative(this->true_Output, this->hidden_layers[no_of_hidden_layers]).get_Row()<<"X"<<
		quadratic_Cost_derivative(this->true_Output, this->hidden_layers[no_of_hidden_layers]).get_Column()<<") and ("<<
		Activation_Function(relu_derivative, (this->Weights[i]*this->hidden_layers[i])+this->Biases[i]).get_Row()<<"X"<<
		Activation_Function(relu_derivative, (this->Weights[i]*this->hidden_layers[i])+this->Biases[i]).get_Column()<<")"<<std::endl;

		//Problem Here
		this->layers_error[i].resize(this->hidden_layers_vector[i], this->no_of_samples);
		this->layers_error[i]=(i==(this->no_of_hidden_layers-1))? quadratic_Cost_derivative(this->true_Output, this->hidden_layers[no_of_hidden_layers]).hadamard_Product(
							Activation_Function(sigmoid_derivative, (this->Weights[i]*this->hidden_layers[i])+this->Biases[i])): 
							quadratic_Cost_derivative(this->true_Output, this->hidden_layers[no_of_hidden_layers]).hadamard_Product(
							Activation_Function(relu_derivative, (this->Weights[i]*this->hidden_layers[i])+this->Biases[i]));

		i==0? this->Weight_gradient[i].resize(this->hidden_layers_vector[i], this->no_of_features):	
			this->Weight_gradient[i].resize(this->hidden_layers_vector[i], this->hidden_layers_vector[i]); 
		this->Weight_gradient[i]=this->layers_error[i]*this->hidden_layers[i].Transpose();

		this->Bias_gradient[i].resize(this->hidden_layers_vector[i], this->no_of_samples);
		this->Bias_gradient[i]=this->layers_error[i];

		this->Weights[i]=this->Weights[i]-(this->Weight_gradient[i]*this->alpha);
		this->Biases[i]=this->Biases[i]-(this->Bias_gradient[i]*this->alpha);

		this->Weights[i].print();
		std::cout<<".....weight"<<i<<" after Optimization......."<<std::endl;
		this->Biases[i].print();
		std::cout<<".....bias"<<i<<" after Optimization......."<<std::endl;
	}						
	
}

Matrix<double> NeuralNetwork::Predict(Matrix<double> X_test){
	this->hidden_layers[0]=X_test.Transpose();
	forward_Propagation();
	return this->hidden_layers[no_of_hidden_layers];
}

#endif