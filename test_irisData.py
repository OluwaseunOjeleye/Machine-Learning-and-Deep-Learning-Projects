# Scatterplot Matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

def plot():
	# Plotting scattered graph for the sepal length and sepal width
	iris = load_iris()
	X = iris.data[:, :2]  # we only take the first two features.
	y = iris.target

	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

	plt.figure(2, figsize=(8, 6))
	plt.clf()

	# Plot the training points
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
	            edgecolor='k')
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')

	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())

	# To getter a better understanding of interaction of the dimensions
	# plot the first three PCA dimensions
	fig = plt.figure(1, figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)
	X_reduced = PCA(n_components=3).fit_transform(iris.data)
	ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
	           cmap=plt.cm.Set1, edgecolor='k', s=40)
	ax.set_title("First three PCA directions")
	ax.set_xlabel("1st eigenvector")
	ax.w_xaxis.set_ticklabels([])
	ax.set_ylabel("2nd eigenvector")
	ax.w_yaxis.set_ticklabels([])
	ax.set_zlabel("3rd eigenvector")
	ax.w_zaxis.set_ticklabels([])

	plt.show()

##plot()

##Function for changing the the bumber labels to flower names
def changeOutputToLabel(p):
	l=[]
	for i in range(len(p)):
		if(p[i]==0):
			l.append("Iris Setosa")
		elif(p[i]==1):
			l.append("Iris Versicolour")
		else:
			l.append("Iris Virginica")
	return l

##Importing datas from iris
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data   ## x- input has 4 data variables-we're working with 4D
y=iris.target  ##y- output for each x data set
##Splitting the dataset into Training data and testing data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)


#Training Clasifier Model Using the K-Nearest Neighbor Model I built
import KNN
clf=KNN.K_NearestNeighbor(3)
#fitting training datas
clf.fit(x_train,y_train)

##Prediction-classifying a new data
predictions=clf.predict(x_test)
print ("Predictions of Testing data is: ",predictions) ##This printout must the same with the next print out

##Testing the accuracy of the data by comparing the predicted output to the testing output
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predictions)
print("Model Accuracy is: ",accuracy)
#################################################################################################################

