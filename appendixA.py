import numpy as np
from pysr import PySRSequenceRegressor

# Generate some data
X = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
for i in range(3, 20):
    X.append([
        np.cos(0.832 + X[i-1][2]) + 0.7 * np.exp(X[i-3][2]),
        ((X[i-2][1])/3.2) + X[i-3][0],
        np.log10(X[i-1][0] ** 2 + X[i-2][1])
    ])
X = np.array(X)
model = PySRSequenceRegressor(
    recursive_history_length=3,
    binary_operators=["+", "*", "-", "/", "^"],
    unary_operators=["sin"],
    niterations=100
)
print(X)
model.fit(X)
print(model.equations_)
print(model.latex())

# interpolation
pred = model.predict(X, num_predictions=18)[:-1]
print("Interpolation MSE: ", np.mean((pred - X[3:]) ** 2))
print("Interpolation MPE: ", np.mean((np.abs(pred - X[3:]))/X[3:]) * 100)

# extrapolation
true = X[-3:].tolist()
for i in range(18):
    true.append([
        np.cos(0.832 + true[-1][2]) + 0.7 * np.exp(true[-3][2]),
        ((true[-2][1])/3.2) + true[-3][0],
        np.log10(true[-1][0] ** 2 + true[-2][1])
    ])
true = np.asarray(true)[3:]
pred = model.predict(X, num_predictions=35)[17:]
print("Extrapolation MSE: ", np.mean((pred-true) ** 2))
print("Extrapolation MPE: ", np.mean((np.abs(pred - true))/true) * 100)