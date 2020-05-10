# Using the model

## Import and instantiate the class

```
code:   clf=K_NearestNeighborClassifier(K_Value)

```
         

## Fit the classifier using the training set
```
code:   clf.fit(x_train,y_train)
Note:	k is no of neighbors between 1 to 9
```

## Make Prediction:

```
code:   predictions=clf.predict(x_test)
		print ("Predictions of Testing data is: ",predictions)
```