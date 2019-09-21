from sklearn.datasets import load_boston
boston=load_boston()
x=boston.data   ## x- input has 4 data variables-we're working with 4D
y=boston.target  ##y- output for each x data set



##Splitting the dataset into Training data and testing data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)

### Using Model
import KNN
clf=KNN.K_NearestNeighborRegression(1)

### fit the classifier using the training set
clf.fit(x_train,y_train)

### Prediction
predictions=clf.predict(x_test)
print ("Predictions of Testing data is: ",predictions)

### Accuracy Test
##Testing the accuracy of the data by comparing the predicted output to the testing output
from sklearn.metrics import mean_squared_error
accuracy=mean_squared_error(y_test,predictions)
print("Model Accuracy is: ",accuracy)