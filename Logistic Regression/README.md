# Using the model

## Import and instantiate the class

```
code:   clf=LogisticRegression()
```
         

## Fit the classifier using the training set
```
code:   clf.fit(X_train,y_train)
```

## Make Predictions:

```
code:   predictions=clf.predict(X_test)
		rounded_predictions=["{:.2f}".format(value) for value in predictions]
		print("Prediction is: ",rounded_predictions)
```