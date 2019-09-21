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

def Accuracy(predictions, y_test):
    correct=0
    wrong=0
    Predict=[]
    for  i in range (len(predictions)):
        if predictions[i]==-1:
            Predict.append(1)
        else:
            Predict.append(0)

    for i in range (len(y_test)):
        if Predict[i] == y_test[i]:
            #print ('Pred: {0} Actual:{1}'.format(Predict[i], y_test[i]))
            correct=correct+1
        else:
            #print('wrong prediction')
            #print ('Pred: {0} Actual:{1}'.format(Predict[i], y_test[i]))
            wrong=wrong+1
    return (correct/(correct+wrong))*100

##Testing the Model
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer.feature_names)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)

#Training Model
import LANL_SVM as SVM
clf=SVM.SVM()
clf.fit(X_train,y_train)

##R_square=clf.R_squared()
##print("R_Square is: ", R_square)

predictions=clf.predict(X_test)
##print("Prediction is: ", predictions)
##print("Real Prediction is: ",y_test)
rounded_predictions=["{:.2f}".format(value) for value in predictions]
print("Prediction: ",rounded_predictions)
print("Real Prediction is: ",y_test)
print("Accuracy: ", Accuracy(predictions, y_test))
