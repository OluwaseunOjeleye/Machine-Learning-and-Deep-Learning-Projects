import numpy as np
import pandas as pd
from numpy.linalg import inv

class LinearRegression():

	##Fitting data on a graph
	def fit(self,X_train,Y_train):
		self.X_train=X_train
		self.Y_train=Y_train

	##Implementing Linear Regression
	def predict(self,X_test):
		### finding the predict with the regression line equation
		predict=self.RegressionlineEqn(X_test)
		return predict

	def RegressionlineEqn(self, X):
		Y=self.Y_train
		##Finding The Prediction with Y=(Transpose(W)).X
		W=self.W_Matrix()
		trans_W=W.transpose()

		##Generating ones on the first column of Matrix X
		int = np.ones(shape=X.shape[0])[..., None]# create vector of ones...
		X = np.concatenate((int, X), 1)

		Y_prediction=X.dot(trans_W)
		return Y_prediction

		##Finding W with W=(Inverse(Transpose(X).X)).((Transpose(X)).Y)
	def W_Matrix(self):
		X=self.X_train
		Y=self.Y_train

		##Generating ones on the first column of Matrix X
		int = np.ones(shape=Y.shape)[..., None]# create vector of ones...
		X = np.concatenate((int, X), 1)

		trans_X=X.transpose()
		mult1=trans_X.dot(X)
		inv_mult1=inv(mult1)

		mult2=trans_X.dot(Y)

		W=inv_mult1.dot(mult2)
		rounded_W=["{:.2f}".format(value) for value in W]
		print("W is: ",rounded_W)
		return W

##################################################################

##Testing the Model
import mglearn
X, y = mglearn.datasets.load_extended_boston()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

##plot(x_train,y_train)

clf=LinearRegression()

clf.fit(X_train,y_train)


##print(clf.W_Matrix())
##R_square=clf.R_squared()
##print("R_Square is: ", R_square)

predictions=clf.predict(X_test)
rounded_predictions=["{:.2f}".format(value) for value in predictions]
print("Prediction is: ",rounded_predictions)
print("Real Prediction is: ",y_test)