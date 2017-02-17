from LinearRegression import LinearRegression, parse_data, create_output_file
import numpy as np
import sys

algo = str(sys.argv[1])
print("Running "+algo)

linreg = LinearRegression()

data = parse_data('train.csv', train=False)

W_temp = []
n_iter = 1000
for i in range(n_iter):
    np.random.shuffle(data)

    X_train = data[0:280, 0:len(data[0])-1]
    y_train = data[0:280, len(data[0])-1:]

    if algo == "gradient_descent":
        W_temp.append(linreg.gradient_descent(X_train, y_train))
    else:
        W_temp.append(linreg.closed_form_solution(X_train, y_train))

W_temp = np.array(W_temp)
W = []
for j in range(W_temp.shape[1]):
    sm = 0
    for i in range(n_iter):
        sm += W_temp[i][j][0]
    sm /= n_iter
    W.append(sm)

W = np.array(W)
W = W.reshape((len(W), 1))

X_test, y_pred = parse_data('test.csv', train=False), []
X_test = linreg.normalize_data(X_test)
X_test = linreg.add_features(X_test)
X_test = np.insert(X_test, 0, 1, axis=1)

y_pred = []
for x in X_test:
    y_pred.append(linreg.predict(x, W)[0])

create_output_file(y_pred)
