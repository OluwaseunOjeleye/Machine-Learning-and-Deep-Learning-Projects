#include "include/NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<int> no_of_layers, const std::string h_Layer_Activation, 
						const std::string output_Layer_Activation): DERIVATIVE("_derivative"){
	this->no_of_samples=1;	//Stochastic Gradient Descent is used i.e batch size=1
	this->no_of_features=0;
	this->hidden_layers_vector=no_of_layers;
	this->no_of_hidden_layers=this->hidden_layers_vector.size()+1;	//+1 including output
	this->hidden_layers_Activation=h_Layer_Activation;
	this->outputs_activation=output_Layer_Activation;
	this->alpha=0.9;
	this->is_trained=false;
}

NeuralNetwork::NeuralNetwork(const std::string filename): DERIVATIVE("_derivative"){
	try{
		std::ifstream inData(filename, std::ios::in);
		if(!inData) throw std::ios::failure( "Error opening file!" );

		std::string line; 
	    int weight=1, bias=1, h_layers=1;
	    while (getline(inData, line)){
	    	if(line==("Hidden Layer "+std::to_string(h_layers))){
	    		int layers;
	    		inData>>layers;
	    		this->hidden_layers_vector.push_back(layers);
	    		h_layers++;
	    	}
	    	else if(line=="Hidden Layers Activation Function:"){
	    		inData>>this->hidden_layers_Activation;
	    	}
	    	else if(line=="Output Layer Activation Function:"){
	    		inData>>this->outputs_activation;
	    	}
	    	else if(line=="Learning Rate:"){
	    		inData>>this->alpha;
	    	}
	    	else if(line==("Weight "+std::to_string(weight))){
	    		Matrix<double> matrix;
	    		inData>>matrix;
	    		this->Weights.push_back(matrix);
	    		weight++;
	    	}
	    	else if(line==("Bias "+std::to_string(bias))){
	    		Matrix<double> matrix;
	    		inData>>matrix;
	    		this->Biases.push_back(matrix);
	    		bias++;
	    	}
	    	else{	}
	    }

		inData.close();

		this->no_of_samples=1;	//Stochastic Gradient Descent is used i.e batch size=1
		this->no_of_features=this->Weights[0].get_Column();
		this->no_of_hidden_layers=this->hidden_layers_vector.size();// including output
		this->is_trained=true;

		//Initiaizing Hidden Layer Matrices
		this->hidden_layers.resize(this->no_of_hidden_layers+1);	//+1 including input
		this->hidden_layers[0].resize(this->no_of_features, this->no_of_samples);
		for(int i=1; i<=this->no_of_hidden_layers; i++){
			this->hidden_layers[i].resize(this->hidden_layers_vector[i-1], this->no_of_samples);	//h1 to output(hn)
		}
	}
	catch (const std::exception& e){
		std::cout<<e.what()<<std::endl;
	}
	
}


NeuralNetwork::~NeuralNetwork(){

}

void NeuralNetwork::Train(std::vector<double> X, std::vector<double> Y, double alpha){

	Matrix<double> X_train(X);
	Matrix<double> Y_train(Y);

	this->alpha=alpha;
	//Initiaizing Hidden Layer Matrices
	this->hidden_layers.resize(this->no_of_hidden_layers+1);	//+1 including input
	this->hidden_layers[0]=X_train.Transpose();	//h0=transpose(X) and its dimension is no_of_feature X no_of_sample
	this->hidden_layers_vector.push_back(Y_train.get_Column());
	for(int i=1; i<=this->no_of_hidden_layers; i++){
		this->hidden_layers[i].resize(this->hidden_layers_vector[i-1], this->no_of_samples);	//h1 to output(hn)
	}
	
	if(!this->is_trained)initialize_Parameters();
	forward_Propagation();
	this->true_Output=Y_train.Transpose();
	//compute_Cost();
	backward_Propagation();
}


void NeuralNetwork::initialize_Parameters(){
	this->no_of_features=hidden_layers[0].get_Row();
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
	this->is_trained=true;
}

void NeuralNetwork::forward_Propagation(){
	for(int i=1; i<=this->no_of_hidden_layers; i++){
		this->hidden_layers[i]=(i!=this->no_of_hidden_layers)?Activation_Function(hidden_layers_Activation, (this->Weights[i-1]*this->hidden_layers[i-1])+this->Biases[i-1]):
			this->hidden_layers[i]=Activation_Function(outputs_activation, (this->Weights[i-1]*this->hidden_layers[i-1])+this->Biases[i-1]);
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
		
		//errors at each layers computation
		this->layers_error[i].resize(this->hidden_layers_vector[i], this->no_of_samples);
		this->layers_error[i]=(i==(this->no_of_hidden_layers-1))? quadratic_Cost_derivative(this->true_Output, this->hidden_layers[no_of_hidden_layers]).hadamard_Product(
							Activation_Function(outputs_activation+DERIVATIVE, (this->Weights[i]*this->hidden_layers[i])+this->Biases[i])): 
							(this->Weights[i+1].Transpose()*this->layers_error[i+1]).hadamard_Product(
							Activation_Function(hidden_layers_Activation+DERIVATIVE, (this->Weights[i]*this->hidden_layers[i])+this->Biases[i]));

		//gradient of weights computation
		i==0? this->Weight_gradient[i].resize(this->hidden_layers_vector[i], this->no_of_features):	
			this->Weight_gradient[i].resize(this->hidden_layers_vector[i], this->hidden_layers_vector[i]); 
		this->Weight_gradient[i]=this->layers_error[i]*this->hidden_layers[i].Transpose();



		//gradients of Biases computation
		this->Bias_gradient[i].resize(this->hidden_layers_vector[i], this->no_of_samples);
		this->Bias_gradient[i]=this->layers_error[i];

		//Updating Weight and Bias
		this->Weights[i]=this->Weights[i]-(this->Weight_gradient[i]*this->alpha);
		this->Biases[i]=this->Biases[i]-(this->Bias_gradient[i]*this->alpha);
	}					
}

Matrix<double> NeuralNetwork::Predict(std::vector<double> X){
	Matrix<double> X_test(X);
	this->hidden_layers[0]=X_test.Transpose();
	forward_Propagation();
	return this->hidden_layers[no_of_hidden_layers];
}

void NeuralNetwork::save_NeuralNetwork(const std::string filename){
	std::ofstream outData;
    outData.open(filename);
 
    for(int i=0; i<this->no_of_hidden_layers; i++){
    	outData<<"Hidden Layer "<<i+1<<std::endl;
    	outData<<this->hidden_layers_vector[i]<<std::endl;//Including output layer
    }
    outData<<std::endl;
	
	outData<<"Hidden Layers Activation Function:"<<std::endl;
	outData<<this->hidden_layers_Activation<<std::endl<<std::endl;
	outData<<"Output Layer Activation Function:"<<std::endl;
	outData<<this->outputs_activation<<std::endl<<std::endl;
	outData<<"Learning Rate:"<<std::endl;
	outData<<this->alpha<<std::endl<<std::endl;

    //Saving all weights
    for(int i=0; i<this->no_of_hidden_layers; i++){
    	outData<<"Weight "<<i+1<<std::endl;
    	outData<<this->Weights[i];
    	outData<<std::endl;
    }

    //Saving all biases
    for(int i=0; i<this->no_of_hidden_layers; i++){
    	outData<<"Bias "<<i+1<<std::endl;
    	outData<<this->Biases[i];
    	outData<<std::endl;
    }
    outData.close();
	
}
