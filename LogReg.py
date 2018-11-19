import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def sigma(x):
    return 1 / (1 + np.exp(x))

def predict(X,w):
    n_datapoints = X.shape[0]
    pred = np.zeros(n_datapoints)       # initialize prediction vector
    for i in range(n_datapoints):
        x=sigma(np.dot(X[i],w))         # use w for prediction
        if x>=0.5:
            pred[i]=0
        else:pred[i]=1
    return pred

def accuracy(X, y, w):
    y_pred = predict(X, w)
    n_tr=X.shape[0]
    correct_pred=0
    for i in range(n_tr):
        if y_pred[i]==y[i]:correct_pred+=1
    return (float(correct_pred)/n_tr)

def logistic_reg(X_tr, X_ts,y_ts,y_tr,lr):
    #perform gradient descent
    n_vars = X_tr.shape[1]  # number of variables
    w = np.zeros(n_vars)    # initialize parameter w
    tolerance = 0.01        # tolerance for stopping
    iter = 0                # iteration counter
    Iteration,test_accuracy,train_accuracy=[],[],[]
    max_iter = 1000      # maximum iteration

    while (True):
        iter += 1

        # calculate gradient
        grad = np.zeros(n_vars) # initialize gradient
        with np.errstate(all ='raise'):
            try:
                for i in range(len(X_tr)):
                    a=0
                    a=np.dot(w,X_tr[i]) #(w,xi)
                    x=y_tr[i]-(np.exp(a)/(1+np.exp(a))) #actual-predicted
                    grad+=x*X_tr[i]
                w_new = w + lr*grad
            except FloatingPointError:
                break
        if iter%50 == 0:
                #print('Iteration: {0}, mean abs gradient = {1}'.format(iter, np.mean(np.abs(grad))))
            Iteration.append(iter)
            test_accuracy.append(accuracy(X_ts, y_ts, w))
            train_accuracy.append(accuracy(X_tr, y_tr, w))
        # stopping criteria and perform update if not stopping
        if (np.mean(np.abs(grad)) < tolerance):
            w = w_new
            break
        else :
            w = w_new

        if (iter >= max_iter):
            break

    return test_accuracy, train_accuracy,Iteration

def Regularization(X_tr, X_ts,y_ts,y_tr,lr,lmbda):
        #perform gradient descent
        n_vars = X_tr.shape[1]  # number of variables
        w = np.zeros(n_vars)    # initialize parameter w
        tolerance = 0.01        # tolerance for stopping

        iter = 0                # iteration counter
        max_iter = 1000        # maximum iteration
        while (True):
            iter += 1

            # calculate gradient
            grad = np.zeros(n_vars) # initialize gradient
            with np.errstate(all ='raise'):
                try:
                    for i in range(len(X_tr)):
                        a=0
                        a=np.dot(w,X_tr[i]) #(w,xi)
                        x=y_tr[i]-(np.exp(a)/(1+np.exp(a))) #rt-yt
                        grad+=x*X_tr[i]
                    w_new = w + lr*(grad-(lmbda*w)) # subtracting Regularization term
                except FloatingPointError:
                    break

            # stopping criteria and perform update if not stopping
            if (np.mean(np.abs(grad)) < tolerance):
                w = w_new
                break
            else :
                w = w_new

            if (iter >= max_iter):
                break

        test_accuracy=accuracy(X_ts, y_ts, w)
        train_accuracy=accuracy(X_tr, y_tr, w)
        return test_accuracy, train_accuracy

def plot(title,xlabel,ylabel,x,y,y1):
        plt.title('Learning Rate:'+str(title))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y,label='Training accuracy')
        plt.plot(x, y1,label='Testing accuracy' )
        legend = plt.legend(loc='upper left', shadow=True)
        plt.show()

D_tr = genfromtxt('spambasetrain.csv', delimiter = ',')
D_ts = genfromtxt('spambasetest.csv', delimiter = ',')

# construct x and y for training and testing
X_tr = D_tr[: ,: -1]
y_tr = D_tr[: , -1]
X_ts = D_ts[: ,: -1]
y_ts = D_ts[: , -1]

# number of training / testing samples
n_tr = D_tr.shape[0]
n_ts = D_ts.shape[0]

# add 1 as feature
X_tr = np.concatenate((np.ones((n_tr, 1)), X_tr), axis = 1)
X_ts = np.concatenate((np.ones((n_ts, 1)), X_ts), axis = 1)

for i in (1e-2,1e-4,1e-6):
    test_accuracy, train_accuracy ,iteration= logistic_reg(X_tr, X_ts,y_ts,y_tr,i)
    plot(i,'Iteration','Accuracy',iteration,train_accuracy,test_accuracy)

#Regularization
Lmbda,y,y1=[],[],[]
lmbda=[-8,-6,-4,-2,0,2]
k=[-8,-6,-4,-2,0,2]
for lmbda in list(map(lambda x:2**x,lmbda)):
    test_accuracy, train_accuracy=Regularization(X_tr, X_ts,y_ts,y_tr,1e-4,lmbda)
    y.append(test_accuracy)
    y1.append(train_accuracy)
plot('.0001','k','Accuracy',k,y1,y)
