import numpy as np

import matplotlib.pyplot as plt
def plot(X,y):
	plt.figure(2, figsize=(8, 6))
	plt.clf()

	# Plot the training points
	plt.scatter(X,y, c=y, cmap=plt.cm.Set1,
	            edgecolor='k')
	plt.xlabel('x')
	plt.ylabel('y')


	plt.xticks(())
	plt.yticks(())
	plt.show()

class LinearRegression():

	##Fitting data on a graph
	def fit(self,x_train,y_train):
		self.x_train=x_train
		self.y_train=y_train

	##Implementing Linear Regression
	def predict(self,x_test):
		predictions=[]

		### finding the prediction of each data
		for x in x_test:
			### finding the predict with the regression line equation
			predict=self.RegressionlineEqn(x)
			##storing predictions in an array
			predictions.append(predict)
		return predictions

	def R_squared(self):
		x=self.x_train
		y=self.y_train
		m=self.getModelslope()
		b=self.getModelintercept(m)

		print("Slope is: ",m)
		print("Intercept is: ",b)

		## For SE_regressionline/Best fit line
		SE_regressionline=0
		for i in range (len(self.x_train)):
			SE_regressionline+=((y[i]-((m*x[i])+b))**2)
		
		## For SE_y
		var_y=self.cov(y,y)
		SE_y=var_y*(len(y)-1)

		## For r_square
		print("SE_regressionline is: ",SE_regressionline)
		print("SE_y is: ",SE_y)
		r_sqr=1-(SE_regressionline/SE_y)
		return r_sqr


	#######Finding values for the slope and intercept for Regression line to create the model##############
	def RegressionlineEqn(self,x):
		m=self.getModelslope()
		b=self.getModelintercept(m)
		y=(m*x)+b
		return y[0]


	def getModelslope(self):
		
		var_x=self.cov(self.x_train,self.x_train)
		##X=np.stack(,axis=0)
		cov_xy=self.cov(self.x_train,self.y_train)
		slope=cov_xy/var_x
		return slope

	def getModelintercept(self,slope):
		mean_y=self.mean(self.y_train)
		mean_x=self.mean(self.x_train)
		intercept=mean_y-(mean_x*slope)
		return intercept

	def cov(self,x,y):
		c=0
		mean_x=self.mean(x)
		mean_y=self.mean(y)
		for i in range (len(x)):
			c+=((x[i]-mean_x)*(y[i]-mean_y))
		cov=c/(len(x)-1)
		return cov

	def mean(self, list):
		sum=0
		for i in range(len(list)):
			sum+=list[i]
		mean=sum/(len(list))
		return mean

##################################################################
import mglearn
x, y = mglearn.datasets.make_wave(n_samples=60)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

##plot(x_train,y_train)

clf=LinearRegression()

clf.fit(x_train,y_train)

R_square=clf.R_squared()
print("R_Square is: ", R_square)

predictions=clf.predict(x_test)
print("Prediction is: ", predictions)
print("Real Prediction is: ",y_test)


