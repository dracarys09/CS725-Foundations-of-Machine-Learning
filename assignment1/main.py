from LinearRegression import LinearRegression, parse_data, create_output_file
import numpy as np


def predict(x, W):
    predicted_value = np.dot(W, x)
    return predicted_value


X_train, y_train = parse_data('train.csv')
# print(X_train.shape)
# X_train = np.insert(X_train, 0, 1, axis=1)
# W = np.random.randn(len(X_train[0]))
# n_coef = len(W)
# m = len(X_train)
#
# y_pred = [predict(x, W) for x in X_train]
# cost = sum([(y_pred[i] - y_train[i])**2 for i in range(m)])[0]
# grad = [0.0 for _ in range(n_coef)]
# alpha = 0.000001
#
#
# print(cost)
# print(len(grad))
#
# for _ in range(1000):
#     for j in range(n_coef):
#         grad[j] = 1.0/m * sum([(y_pred[i] - y_train[i])*X_train[i][j] for i in range(m)])[0]
#
#     print(grad)
#
#     for j in range(n_coef):
#         W[j] = W[j] - alpha*grad[j]
#
#     y_pred = [predict(x, W) for x in X_train]
#     MSE = sum([(y_pred[i] - y_train[i])**2 for i in range(m)])[0]
#
#     cost = MSE
#
#     print(cost)


linreg = LinearRegression()
alpha = 0.00001
eps = 0.000001
lam = 200000
W_gradient_descent = linreg.gradient_descent(X_train, y_train, alpha, eps, lam)
W_closed_form = linreg.closed_form_solution(X_train, y_train)
X_test, y_pred_gradient_descent, y_pred_closed_form = parse_data('test.csv', train=False), [], []
X_test = np.insert(X_test, 0, 1, axis=1)


for x in X_test:
    y_pred_gradient_descent.append(linreg.predict(x, W_gradient_descent))
    y_pred_closed_form.append(linreg.predict(x, W_closed_form))


print("Gradient Descent W")
print(y_pred_gradient_descent)

print("Closed Form W")
print(y_pred_closed_form)

# create_output_file(y_pred)
