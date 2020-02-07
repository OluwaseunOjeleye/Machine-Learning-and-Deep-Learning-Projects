# Matrix Class

Implementation of matrix class in C++. This class contains matrix operators that are important when implementing machine learning and deep learning models.

### Downloading
Cloning The GitHub Repository

```
git clone 
```
### Methods

####Constructors:
Matrix()

Matrix(row, column): creating (row X column) matrix

Matrix(row, column, number): creating (row X column) matrix initialized with number

Matrix(vector<T> matrix): creating matrix with 1-D vector

Matrix(vector<vector<T>> matrix): creating matrix with 2-D vector

#### Methods:

resize(row, column): resizing matrix

set_Element(row, column, value): set element of matrix

get_Element(row, column): get element of matrix 

get_Row(): get no of rows of matrix

get_Column(): get no of column of matrix

generate_Random_Elements(): set all element of matrix with random numbers

Transpose(): returns transpose of matrix

hadamard_Product(mat2):	returns harmard product of matrix with mat2

print: prints matrix on screen

#### Overloaded Operators:

Assignment operator(=)

Addition operator(+): For Matrix-Matrix and Matrix-Scalar Addition

Subtraction operator(-): For Matrix-Matrix and Matrix-Scalar Subtraction

Multiplication operator(*): For Matrix-Matrix and Matrix-Scalar Multiplication

Division operator(/): For Matrix-Scalar Division

Modulo operator(%): For Matrix-Scalar Modulo

power operator(^): For Matrix-Scalar power

read matrix to file (>>)

write matrix to file (<<)

#### Examples
```
Matrix<int> Mat1(3,3,3);
Matrix<int> Mat2({{1,2,3}, {4,5,6}, {7,8,9}});

Matrix<int> Mat3=Mat1+Mat2;
Matrix<int> Mat3=Mat1+2;
Matrix<int> Mat3=Mat1-Mat2;
Matrix<int> Mat3=Mat1-2;
Matrix<int> Mat3=Mat1*Mat2;
Matrix<int> Mat3=Mat1*2;
Matrix<double> Mat3=Mat1/2;
Matrix<int> Mat3=Mat1^2;

Matrix<int> Mat3=Mat1.Transpose();
Matrix<int> Mat3=Mat1.hadamard_Product(Mat2);
Mat1.print();

outputFile<<Mat2;
inputFile>>Mat3;
```

## Authors
* **Jamiu Oluwaseun Ojeleye** 