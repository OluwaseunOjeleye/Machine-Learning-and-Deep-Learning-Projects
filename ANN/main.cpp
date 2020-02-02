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
	Matrix<double> X_train({{252, 4, 155, 175}, {175, 10, 186, 200}, {82, 131, 230, 100}, {115, 138, 80, 88}});
	//Matrix<double> Y_train({{1,1,0,0}, {0, 0, 1, 1}});
	Matrix<double> Y_train({0, 0, 1, 1});
	std::vector<int> layers={2,1};
    NeuralNetwork Network1(layers);

    Network1.Train(X_train, Y_train, 0.5, "a", "b");
    return 0;
}