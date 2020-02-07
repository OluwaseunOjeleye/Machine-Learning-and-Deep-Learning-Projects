#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <random>

#define R_MIN -1
#define R_MAX 1

#define ASSERT(condition, message){\
    if(!condition){\
        std::cerr<<"Failure in "<<__FILE__<<" line "<<__LINE__<<": "<<message<<std::endl;\
        std::terminate();\
    }\
}

template <typename T>
class Matrix{
    template <typename U> friend std::istream& operator>>(std::istream &filename, Matrix<U> &mat);
    template <typename U> friend void operator<<(std::ostream &filename, Matrix<U> &mat);

    public:
        Matrix();
        Matrix(int row, int column);
        Matrix(int row, int column, T number);
        Matrix(std::vector<T> matrix);
        Matrix(std::vector<std::vector<T>> matrix);
        ~Matrix();

        void resize(int row, int column);
        void set_Element(int row, int column, T value);
        T get_Element(int row, int column) const;
        int get_Row() const;
        int get_Column() const;
        void generate_Random_Elements();

        Matrix operator=(const Matrix &mat1);
        Matrix operator+(const Matrix &mat2) const; //Addition
        Matrix operator+(const T value) const; //Addition scalar
        Matrix operator-(const Matrix &mat2) const; //Subtraction
        Matrix operator-(const T value) const; //Subtraction scalar
        Matrix operator*(const Matrix &mat2) const; //Multiplication
        Matrix operator*(const T value) const; //Multiplication scalar
        Matrix operator/(const T value) const; //Division scalar
        Matrix operator%(const int value) const; //Modulo scalar
        Matrix operator^(const int value) const; //power scalar

        Matrix<T> Transpose();
        Matrix<T> hadamard_Product(const Matrix &mat2)const;
        void print() const;

    private:
        T randomNo(int min, int max);
        std::vector<std::vector<T>> matrix;
        int row;
        int column;        
};

template <typename T> 
Matrix<T>::Matrix(){
    this->row=0;
    this->column=0;
}

template <typename T> 
Matrix<T>::Matrix(int row, int column){
    this->row=row;
    this->column=column;
    this->matrix.resize(this->row, std::vector<T> (this->column));
}

template <typename T> 
Matrix<T>::Matrix(int row, int column, T number){
    this->row=row;
    this->column=column;
    this->matrix.resize(this->row, std::vector<T> (this->column));

    for (int i=0; i<this->row; i++){
        for (int j=0; j<this->column; j++){
            this->matrix[i][j]=number;
        }
    }
}

template <typename T> 
Matrix<T>::Matrix(std::vector<T> matrix){
    this->row=1;
    this->column=matrix.size();
    this->matrix.resize(this->row, std::vector<T> (this->column));

    for (int i=0; i<this->column; i++){
        this->matrix[0][i]=matrix[i];
    }
}


template <typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> matrix){
    this->row=matrix.size();
    this->column=matrix[0].size();
    this->matrix.resize(this->row, std::vector<T> (this->column));

    for (int i=0; i<this->row; i++){
        for (int j=0; j<this->column; j++){
            this->matrix[i][j]=matrix[i][j];
        }
    }
}

template <typename T> 
Matrix<T>::~Matrix(){
    
}

template <typename T>
void Matrix<T>::resize(int row, int column){
    this->row=row;
    this->column=column;
    this->matrix.clear();
    this->matrix.shrink_to_fit();
    this->matrix.resize(this->row, std::vector<T> (this->column));
}


template <typename T> 
void Matrix<T>::set_Element(int row, int column, T value){
    ASSERT(((row<this->row)&&(column<this->column)), "Out Of Bound");
    this->matrix[row][column]=value;
}

template <typename T> 
T Matrix<T>::get_Element(int row, int column) const{
    ASSERT(((row<this->row)&&(column<this->column)), "Out Of Bound");
    return this->matrix[row][column];
}

template <typename T>
int Matrix<T>::get_Row() const{
    return this->row;
}

template <typename T>
int Matrix<T>::get_Column() const{
    return this->column;
}

//Random Number Generating function
template <typename T> 
T Matrix<T>::randomNo(int min, int max){
   /*double x = rand()/static_cast<double>(RAND_MAX); 
   T random = min + static_cast<T>( x * (max - min) );*/
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<T> distr(min, max);
    return distr(generator);
}

//Function for generating random numbers in matrix
template <typename T>
void Matrix<T>::generate_Random_Elements(){
    for (int i=0; i<this->row; i++){
        for (int j=0; j<this->column; j++){
            this->matrix[i][j]=randomNo(R_MIN, R_MAX);  //random number between min and max
        }
    }
}

//assignment operator =
template <typename T> 
Matrix<T> Matrix<T>::operator=(const Matrix &mat1){
    if(this!=&mat1){
        this->row=mat1.row;
        this->column=mat1.column;

        this->matrix.resize(mat1.row, std::vector<T> (mat1.column));

        for(int i=0; i<mat1.row; i++){
            for(int j=0; j<mat1.column; j++){
                this->matrix[i][j]=mat1.matrix[i][j];
            }
        }
    }
    return *this;
}

//Addition Operator
template <typename T> 
Matrix<T> Matrix<T>::operator+(const Matrix &mat2)const{
    ASSERT(((this->row==mat2.row)&&(this->column==mat2.column)), "Cannot add these two matrices");

    Matrix<T> Result(this->row, this->column);
    for(int i=0; i<this->row; i++){
        for(int j=0; j<this->column; j++){
            Result.matrix[i][j]=this->matrix[i][j]+mat2.matrix[i][j];
        }
    }
    return Result;
}

//Addition scalar operator
template <typename T> 
Matrix<T> Matrix<T>::operator+(const T value)const{
    Matrix<T> Result(this->row, this->column);
    for(int i=0; i<this->row; i++){
        for(int j=0; j<this->column; j++){
            Result.matrix[i][j]=this->matrix[i][j]+value;
        }
    }
    return Result;
}

//Subtraction operator
template <typename T> 
Matrix<T> Matrix<T>::operator-(const Matrix &mat2) const{
    ASSERT(((this->row==mat2.row)&&(this->column==mat2.column)), "Cannot subtract these matrices");

    Matrix<T> Result(this->row, this->column);
    for(int i=0; i<this->row; i++){
        for(int j=0; j<this->column; j++){
            Result.matrix[i][j]=this->matrix[i][j]-mat2.matrix[i][j];
        }
    }
    return Result;
}

//Subtraction scalar operator
template <typename T> 
Matrix<T> Matrix<T>::operator-(const T value) const{
    Matrix<T> Result(this->row, this->column);
    for(int i=0; i<this->row; i++){
        for(int j=0; j<this->column; j++){
            Result.matrix[i][j]=this->matrix[i][j]-value;
        }
    }
    return Result;
} 

//Multiplication operator
template <typename T> 
Matrix<T> Matrix<T>::operator*(const Matrix &mat2)const{
    ASSERT((this->column==mat2.row), "Matrix1 column and Matrix2 row are not equal.");

    Matrix<T> Result(this->row, mat2.column);
    for(int i=0; i<this->row; i++){
        for(int j=0; j<mat2.column; j++){
            Result.matrix[i][j]=0; 
            for(int k=0;k<this->column;k++){
                Result.matrix[i][j]=Result.matrix[i][j]+(this->matrix[i][k]*mat2.matrix[k][j]);
            }
        }
    }
    return Result;
}

//Multiplication scalar operator
template <typename T> 
Matrix<T> Matrix<T>::operator*(const T value)const{
    Matrix<T> Result(this->row, this->column);
    for(int i=0; i<this->row; i++){
        for(int j=0; j<this->column; j++){
            Result.matrix[i][j]=this->matrix[i][j]*value;
        }
    }
    return Result;
} 

//Division scalar operator
template <typename T> 
Matrix<T> Matrix<T>::operator/(const T value)const{
    Matrix<T> Result(this->row, this->column);
    for(int i=0; i<this->row; i++){
        for(int j=0; j<this->column; j++){
            Result.matrix[i][j]=this->matrix[i][j]/value;
        }
    }
    return Result;
}

//Modulo scalar operator
template <typename T> 
Matrix<T> Matrix<T>::operator%(const int value)const{
    Matrix<T> Result(this->row, this->column);
    for(int i=0; i<this->row; i++){
        for(int j=0; j<this->column; j++){
            Result.matrix[i][j]=int(this->matrix[i][j])%value;
        }
    }
    return Result;
} 

 //power scalar operator
template <typename T> 
Matrix<T> Matrix<T>::operator^(const int value)const{
    Matrix<T> Result(this->row, this->column);
    for(int i=0; i<this->row; i++){
        for(int j=0; j<this->column; j++){
            Result.matrix[i][j]=pow(this->matrix[i][j], value);
        }
    }
    return Result;
}

//Transpose
template <typename T> 
Matrix<T> Matrix<T>::Transpose(){
    Matrix<T> Result(this->column, this->row);
    for(int i=0; i<this->column; i++){
        for(int j=0; j<this->row; j++){
            Result.matrix[i][j]=this->matrix[j][i];
        }
    }
    return Result;
}

//Hadamard Product Function
template <typename T> 
Matrix<T> Matrix<T>::hadamard_Product(const Matrix &mat2)const{
    ASSERT(((this->row==mat2.row)&&(this->column==mat2.column)), "Cannot perform Hadmard Product on these two matrices");

    Matrix<T> Result(this->row, this->column);
    for(int i=0; i<this->row; i++){
        for(int j=0; j<this->column; j++){
            Result.matrix[i][j]=this->matrix[i][j]*mat2.matrix[i][j];
        }
    }
    return Result;
}

//Print Function
template <typename T> 
void Matrix<T>::print() const{
    for(int i=0; i<this->row; i++){
        for(int j=0; j<this->column; j++){
            std::cout<<this->matrix[i][j]<<"\t";
        }
        std::cout<<std::endl;
    }
    std::cout<<"................."<<std::endl;
}

template <typename T> 
std::istream& operator>>(std::istream &inData, Matrix<T> &mat){
    if(inData){
        inData>>mat.row;
        inData>>mat.column;
        mat.resize(mat.row, mat.column);
        for(int i=0; i<mat.row; i++){
            for(int j=0; j<mat.column; j++){
                inData>>mat.matrix[i][j];
            }
        }
    }
    return inData;
}

template <typename T> 
void operator<<(std::ostream &outData, Matrix<T> &mat){
    outData<<mat.row<<std::endl;
    outData<<mat.column<<std::endl;
    for(int i=0; i<mat.row; i++){
        for(int j=0; j<mat.column; j++){
            outData<<mat.matrix[i][j]<<" ";
        }
        outData<<std::endl;
    }
}

#endif

