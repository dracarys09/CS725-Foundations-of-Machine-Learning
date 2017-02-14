import numpy as np
import csv

class LinearRegression:

    def closed_form_solution(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        X_transpose = np.transpose(X)
        W = np.matmul(np.matmul(np.linalg.inv(
            np.matmul(X_transpose, X)), X_transpose), y)
        return W


    def gradient_descent(self, X, y, alpha=0.001, eps=0.0001, lam=20000):
        X = np.insert(X, 0, 1, axis=1)
        converged = False
        m = X.shape[0]
        W = np.random.randn(len(X[0]))
        n_coef = len(W)

        y_pred = [self.predict(x, W) for x in X]
        cost = sum([(y_pred[i] - y[i])**2 for i in range(m)])[0]
        grad = [0.0 for _ in range(n_coef)]

        itr = 0
        while not converged:
            for j in range(n_coef):
                grad[j] = 1.0/m * sum([(y_pred[i] - y[i])*X[i][j] for i in range(m)])[0]
                grad[j] += (lam/m) * W[j]

            for j in range(n_coef):
                W[j] = W[j] - alpha*grad[j]

            y_pred = [self.predict(x, W) for x in X]
            MSE = sum([(y_pred[i] - y[i])**2 for i in range(m)])[0]
            if abs(cost - MSE) <= eps:
                converged = True

            cost = MSE

            itr += 1
            if(itr == 1000):
                break

        return W

    def predict(self, x, W):
        predicted_value = np.dot(W.T, x)
        return predicted_value


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
