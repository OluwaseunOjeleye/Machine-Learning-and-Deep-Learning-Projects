# Using the model

## Import and instantiate the class

```
code:   clf=SVM()- Using default kernel-linear_kernel
	    clf=SVM(C=regularizing_parameter)
	    clf=SVM(kernel=gaussian_kernel)
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