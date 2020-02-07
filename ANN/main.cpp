#include "include/NeuralNetwork.h"

int main(){
/*
	//AND Network	
	std::vector<std::vector<double>> inputs({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
	std::vector<std::vector<double>> outputs({{0}, {0}, {0}, {1}});
*/

/*
	//XOR AND XNOR Network
	std::vector<std::vector<double>> inputs({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
	std::vector<std::vector<double>> outputs({{0,1}, {1, 0}, {1, 0}, {0, 1}});

	std::vector<int> layers={2,2};
    NeuralNetwork Network(layers, "sigmoid", "sigmoid");

    //Training
    std::cout<<"Training..."<<std::endl;
    for(int i=1; i<=50000; i++){
    	for (int j=0; j<4; j++){
    		Network.Train(inputs[j], outputs[j], 0.7);
    	}
    }
	
	//Testing
	std::cout<<"Predicting..."<<std::endl;
    for (int i=0; i<4; i++){
    	std::cout<<"Prediction "<<i+1<<": "<<std::endl;
    	Network.Predict(inputs[i]).print();
    }

    Network.save_NeuralNetwork("./saved_Parameters/AND_NeuralNetwork");

*/

	NeuralNetwork Net("./saved_Parameters/AND_NeuralNetwork");

	std::vector<std::vector<double>> inputs({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
	std::cout<<"Predicting..."<<std::endl;
	for (int i=0; i<4; i++){
    	std::cout<<"Prediction "<<i<<": "<<std::endl;
    	Net.Predict(inputs[i]).print();
    }

    return 0;
}
