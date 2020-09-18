import numpy as np
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing, model_selection

def mse( y_true, y_pred):
   y_err = y_true - y_pred
   y_sqr = y_err * y_err
   y_sum = np.sum(y_sqr)
   y_mse = y_sum / y_sqr.size
   return y_mse

np.set_printoptions(suppress=True)

data = genfromtxt('sample.csv', delimiter=',', skip_header=1)

X = data[:, 0:6]

y = data[:, 6]

X = preprocessing.scale(X)
print(X.shape)
print(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

mlp = MLPClassifier(random_state=0, hidden_layer_sizes=[100], max_iter=1)

i = 1
cont = True
min_error = 1
while(i <= 2000):
    mlp.partial_fit(X_train, y_train, np.unique(y_train))
    y_pred = mlp.predict(X_train)
    mse_err = mse(y_train, y_pred)
    print("Run: " + str(i) + " MSE: " + str(mse_err))
    i = i + 1

depth = mlp.score(X_test, y_test)
print(depth)