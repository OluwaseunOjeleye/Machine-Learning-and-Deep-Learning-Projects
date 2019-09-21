from scipy.spatial import distance
from scipy import stats

class K_NearestNeighbor(object):
	### method for number of nearest neighbor
	def __init__(self, k):
		self.k=k
		
	##Fitting data on a graph
	def fit(self,x_train,y_train):
		self.x_train=x_train
		self.y_train=y_train

	##Implementing K-Nearest Neighbor to Classify and Train the data
	def predict(self,x_test):
		predictions=[]
		##looping over all testing points
		##node-point-row
		for node in x_test:
			##predicting closest training point's output to this particular testing point 
			predict=self.closest(node)
			##storing prediction each testing points in an array
			predictions.append(predict)
		return predictions

	##Finding the closest training point) to a particular testing point(node) on graph
	def  closest(self, node):
		### storing all distances in an array
		distances=[]
		for i in range(0,len(self.x_train)):
			dist=distance.euclidean(node,self.x_train[i])
			distances.append(dist)

		### sorting disances in an array
		sorted_distances=sorted(distances)

		### storing the indexes of best k shortest distances in an array(example best 5 shortest distances)
		best_K_indexes=[]
		for a in range (self.k):
			for b in range (len(self.x_train)):
				if (sorted_distances[a]==distances[b]):
					best_K_indexes.append(b)
					break
		
		### storing the output of the best k training points()
		label=[]
		for j in range (self.k): 
			output=self.y_train[best_K_indexes[j]]
			label.append(output)

		### finding the mode of the label- The highest occuring Label  
		prediction=stats.mode(label)
		return prediction[0][0]

##End Of Training###################################################################################################################################

### Using Model: 
###import and instantiate the class
###			code: 	clf=K_NearestNeighborClassifier(K_Value)

### fit the classifier using the training set
###			code:	clf.fit(x_train,y_train)
###			Note: k is no of neighbors between 1 to 9

### To make predictions on the test data, we call the predict method. For each data point
### in the test set, this computes its nearest neighbors in the training set and finds the
### most common class among them:
###			code:	predictions=clf.predict(x_test)
###					print ("Predictions of Testing data is: ",predictions)