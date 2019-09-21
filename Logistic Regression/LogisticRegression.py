import numpy as np
import pandas as pd
from numpy.linalg import inv

class LogisticRegression():

	##Fitting data on a graph
	def fit(self,X_train,Y_train):
		self.X_train=X_train
		self.Y_train=Y_train

	##Implementing Logistic Regression
	def predict(self,X_test):
		### finding the predict
		X=X_test
		new_weight=self.Newton_Raphson_Optimization()
		##print("New Weight is...",new_weight)

		trans_new_weight=new_weight.transpose()
		
		##Generating ones on the first column of Matrix X
		int = np.ones(shape=X.shape[0])[..., None]# create vector of ones...
		X = np.concatenate((int, X), 1)
		### Finding Predict Response---which is in Regession form
		reg_predict=X.dot(trans_new_weight)
		### Finding Sigmoid function to give prediction in classification form
		predict=self.sigmoid(reg_predict)
		return predict

	def Concat_all_ones_mat(self,X):
		##Generating ones on the first column of Matrix X
		int = np.ones(shape=X.shape[0])[..., None]# create vector of ones...
		new_X= np.concatenate((int, X), 1)
		return new_X

	##Finding Initial Weight
	def init_W_Matrix(self):
		X=self.Concat_all_ones_mat(self.X_train)
		Y=self.Y_train

		trans_X=X.transpose()
		mult1=trans_X.dot(X)
		inv_mult1=inv(mult1)
		mult2=trans_X.dot(Y)
		##Finding Weight
		W=inv_mult1.dot(mult2)
		##rounded_W=["{:.2f}".format(value) for value in W]
		##print("W is: ",rounded_W)
		##print("Initial Weight is...",W)
		return W

	def init_response(self):
		X=self.Concat_all_ones_mat(self.X_train)
		W=self.init_W_Matrix()
		trans_W=W.transpose()
		### Finding Initial Response
		Y=X.dot(trans_W)
		return Y

	##Sigmoid Function for Response
	def sigmoid(self,Y):
		alpha=[]
		for i in range (len(Y)):
			sig=1/(1+(np.exp(-1*Y[i])))
			alpha.append(sig)
		##Returning transpose of c_alpha-(n row*1 column)
		c_alpha=(np.array(alpha)).transpose()
		return c_alpha

	def gradient(self):
		init_response=self.init_response()
		alpha=self.sigmoid(init_response)
		A=self.Concat_all_ones_mat(self.X_train)
		Y=self.Y_train
		grad=A.transpose().dot(alpha-Y)
		return grad

	def alpha_DiagonalMatrix(self):
		b=[]
		init_response=self.init_response()
		alpha=self.sigmoid(init_response)
		for i in range (len(alpha)):
			b.append([])
			for j in range (len(alpha)):
				if(i==j):
					b[i].append(alpha[i]*(1-alpha[i]))
				else:
					b[i].append(0)
		B=(np.array(b))
		return B

	def Hessian(self):
		A=self.Concat_all_ones_mat(self.X_train)
		B=self.alpha_DiagonalMatrix()
		Hessian=A.transpose().dot(B.dot(A))
		return Hessian

	def Newton_Raphson_Optimization(self):
		Hessian=self.Hessian()
		gradient=self.gradient()
		W_old=self.init_W_Matrix()
		W_new=W_old-((inv(Hessian)).dot(gradient))
		return W_new
##End of Training################################################################

### Using Model: 
###import and instantiate the class
###			code: 	clf=LogisticRegression()

### fit the classifier using the training set
###			code:	clf.fit(X_train,y_train)

### To make predictions on the test data, we call the predict method.
###			code:	predictions=clf.predict(X_test)
###					rounded_predictions=["{:.2f}".format(value) for value in predictions]
###					print("Prediction is: ",rounded_predictions)