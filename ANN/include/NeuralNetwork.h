#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_
#include "activations.h"
#include "cost.h"
#include <string>


//Neural Network with Stochastic Gradient Descent as Optimizer
class NeuralNetwork{
	public:
		NeuralNetwork(const std::vector<int> no_of_layers, const std::string h_Layer_Activation, 
						const std::string output_Layer_Activation);//excluding input layers
		NeuralNetwork(const std::string filename);
		~NeuralNetwork();

		void Train(std::vector<double> X, std::vector<double> Y, double alpha);
		Matrix<double> Predict(std::vector<double> X);
		void save_NeuralNetwork(const std::string filename);

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

		std::string hidden_layers_Activation; //Activation function for hidden layers
		std::string outputs_activation; //Activation function function for outputs
		const std::string DERIVATIVE;

		bool is_trained;
};

#endif