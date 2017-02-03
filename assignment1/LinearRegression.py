import argparse
import numpy as np
import csv

parser = argparse.ArgumentParser(description=("Implementation of Linear"
                                              "Regression using Gradient"
                                              "Descent algorithm."))

parser.add_argument("dataset_dir", help="Path of directory containing dataset")
args = parser.parse_args()


class LinearRegression:

    def closed_form_solution(self, X, y):
        self.X = X
        self.y = y
        self.n_coef = len(X[0]) if len(X) > 0 else 0

        X = np.insert(X, 0, 1, axis=1)

        X_transpose = np.transpose(X)
        self.W = np.matmul(np.matmul(np.linalg.inv(
            np.matmul(X_transpose, X)), X_transpose), y)
        self.intercept = self.W[0]
        self.W = self.W[1:]

        return self.intercept, self.W

    def predict(self, x):
        predicted_value = np.dot(self.W[:, 0], x) + self.intercept[0]
        return predicted_value


def parse_data(fname, train=True):
    with open(args.dataset_dir+fname, 'r') as f:
        headers = f.readline().split(',')
        data = np.loadtxt(f, delimiter=',', usecols=range(1, len(headers)))

    if train == True:
        X = data[:, 0:len(data[0])-1]
        y = data[:, len(data[0])-1:]
        return X, y
    else:
        return data


X_train, y_train = parse_data('train.csv')

linreg = LinearRegression()
intercept_, coef_ = linreg.closed_form_solution(X_train, y_train)
X_test, y_pred = parse_data('test.csv', train=False), []
for x in X_test:
    y_pred.append(linreg.predict(x))

with open('output.csv', 'w') as f:
    f.write('ID,MEDV\n')
    i = 0
    for val in y_pred:
        f.write(str(i))
        f.write(',')
        f.write(str(val))
        f.write('\n')
        i += 1
