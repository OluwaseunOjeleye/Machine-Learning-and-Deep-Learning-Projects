import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

class SVM(object):
    ##Accepting regularizing parameter C
    def __init__(self, C=None):
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        ##I made X_train,y_train,n_samples,n_features private variables 
        ##since i will be using them in other methods within the class
        self.X_train=X
        self.y_train=y
        ##Assigning number of rows/dataset samples to n_samples and 
        ##Assigning number of features to n_features 
        self.n_samples, self.n_features = (self.X_train).shape

        # Gram matrix for linear SVM
        self.K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                ## Since We are working with linear SVM, K=X[i].X[j]
                self.K[i,j] = np.dot(X[i], X[j])

    ##Finding alpha using Parameter C,X_train,y_train and n_samples
    def find_Lagrange_Multipliers(self):
        P = cvxopt.matrix(np.outer(self.y_train, self.y_train) * self.K)
        q = cvxopt.matrix(np.ones(self.n_samples) * -1)
        A = cvxopt.matrix(self.y_train, (1, self.n_samples),'d')
        b = cvxopt.matrix(0.0)

        ##If no Regulaizing parameter C
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(self.n_samples) * -1))
            h = cvxopt.matrix(np.zeros(self.n_samples))
        #
        else:
            tmp1 = np.diag(np.ones(self.n_samples) * -1)
            tmp2 = np.identity(self.n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(self.n_samples)
            tmp2 = np.ones(self.n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha= np.ravel(solution['x'])
        ## Making Alpha a private variable instead of returning it so i don't need to call it in other methods
        return alpha

    ##Finding Intercept,b
    def find_intercept(self):
        a=self.find_Lagrange_Multipliers()
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        alp = a[sv]
        sv_y = self.y_train[sv]
        ##Print number of Support vectors out of total points on graph
        print("%d support vectors out of %d points" % (len(alp), self.n_samples))

        # Intercept
        b= 0
        for n in range(len(alp)):
            b+= sv_y[n]
            b-= np.sum(alp * sv_y * self.K[ind[n],sv])

        b /= len(alp)
        return b

    ## finding Weight, W
    def find_Weight(self):
        a=self.find_Lagrange_Multipliers()
        sv = a > 1e-5
        alp = a[sv]
        sv_y = self.y_train[sv]
        sv_X = self.X_train[sv]
        # Weight vector
        w = np.zeros(self.n_features)
        for n in range(len(alp)):
            w += alp[n] * sv_y[n] * sv_X[n]

        return w

    ## Predicting Response/Output for X_test
    def predict(self, X_test):
        b=self.find_intercept()
        W=self.find_Weight()
        predict=np.sign(np.dot(X_test, W) + b)
        return predict

##### End of training..............................................................

### Using Model: 
###import and instantiate the class
###         code:   clf=SVM()
###                 clf=SVM(C=regularizing_parameter)

### fit the classifier using the training set
###         code:   clf.fit(X_train,y_train)

### To make predictions on the test data, we call the predict method.
###         code:   predictions=clf.predict(X_test)
###                 rounded_predictions=["{:.2f}".format(value) for value in predictions]
###                 print("Prediction is: ",rounded_predictions)