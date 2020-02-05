#include "include/NeuralNetwork.h"

int main(){
	//std::cout<<relu_derivative(5)<<std::endl;
	/*int count=0;
	Matrix<double> Mat1(3,3); 
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			Mat1.set_element(i, j, ++count);
		}
	}
	
    Matrix<double> Mat2(3,3,2.56);

    Mat1.print();
    Mat2.print();

    Matrix<double> Mat3=Activation_Function(relu, Mat2);
    Mat3.print();*/

	/*std::vector<std::vector<double>> inputs({{252, 4, 155, 175}, {175, 10, 186, 200}, {82, 131, 230, 100}, {115, 138, 80, 88}});
	std::vector<std::vector<double>> outputs({{1,0}, {1,0}, {0,1}, {0,1}});
*/
	/*std::vector<std::vector<double>> inputs({{252, 4, 155, 175}, {175, 10, 186, 200}, {82, 131, 230, 100}, {115, 138, 80, 88}});
	std::vector<std::vector<double>> outputs({{1,0}, {1,0}, {0,1}, {0,1}});*/

	std::vector<std::vector<double>> inputs({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
	std::vector<std::vector<double>> outputs({{0}, {1}, {1}, {0}});

	/*std::vector<std::vector<double>> inputs({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
	std::vector<std::vector<double>> outputs({{0,1}, {1, 0}, {1, 0}, {0, 1}});
*/
	std::vector<int> layers={2,2,1};
    NeuralNetwork Network1(layers);

    
    for(int i=1; i<=20; i++){
    	for (int j=0; j<4; j++){
    		Matrix<double> X_train(inputs[j]);
    		Matrix<double> Y_train(outputs[j]);
    		Network1.Train(X_train, Y_train, 0.5, "sigmoid", "sigmoid");
    	}
    }

    for (int i=0; i<4; i++){
    	std::cout<<"Prediction "<<i<<": "<<std::endl;
    	Matrix<double> X_train(inputs[i]);
    	Network1.Predict(X_train).print();
    }

    Network1.printWeight();
    



    return 0;
}