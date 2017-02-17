import numpy as np
import csv

class LinearRegression:

    def closed_form_solution(self, X, y, lam=0.25, ridge=True):
        X = self.normalize_data(X)
        X = self.add_features(X)
        X = np.insert(X, 0, 1, axis=1)
        X_transpose = np.transpose(X)
        if ridge:
            W = np.dot(np.dot(np.linalg.inv(
                np.dot(X_transpose, X) + lam*np.identity(len(X_transpose))), X_transpose), y)
        else:
            W = np.dot(np.dot(np.linalg.inv(
                np.dot(X_transpose, X)), X_transpose), y)

        return W


    def gradient_descent(self, X, y, alpha=0.001, eps=0.000001, lam=0.001, n_iter=1000):
        X = self.normalize_data(X)
        X = self.add_features(X)
        X = np.insert(X, 0, 1, axis=1)
        converged = False
        m = X.shape[0]
        W = np.random.randn(len(X[0]))
        n_coef = len(W)
        W = W[:, np.newaxis]

        temp = np.dot(X, W) - y
        cost = np.dot(temp.T, temp)[0]
        itr = 0
        while not converged:
            delta_W = np.dot(X.T, (np.dot(X, W) - y)) + lam*W
            delta_W /= m
            W = W - alpha*delta_W

            temp = np.dot(X, W) - y
            MSE = np.dot(temp.T, temp)[0]

            if abs(cost - MSE) <= eps:
                converged = True

            cost = MSE

            itr += 1
            if(itr == n_iter):
                break

        return W

    def add_features(self, X):
        n_cols = X.shape[1]
        for j in range(n_cols):
            col = X[:,j]
            X = np.column_stack((X, np.square(col)))
            X = np.column_stack((X, np.sqrt(np.abs(col))))
            # X = np.column_stack((X, np.square(col)*np.square(col)))
            # X = np.column_stack((X, np.sqrt(np.abs(col)*np.sqrt(np.abs(col)))))
        return X

    def predict(self, x, W):
        predicted_value = np.dot(x, W)
        return predicted_value


    def normalize_data(self, X):
        return (X - X.mean(axis=0))/X.std(axis=0)



def parse_data(fname, train=True):
    with open('./data/'+fname, 'r') as f:
        headers = f.readline().split(',')
        data = np.loadtxt(f, delimiter=',', usecols=range(1, len(headers)))

    if train == True:
        X = data[:, 0:len(data[0])-1]
        y = data[:, len(data[0])-1:]
        return X, y
    else:
        return data

def create_output_file(predicted_values):
    with open('output.csv', 'w') as f:
        f.write('ID,MEDV\n')
        i = 0
        for val in predicted_values:
            f.write(str(i))
            f.write(',')
            f.write(str(val))
            f.write('\n')
            i += 1
