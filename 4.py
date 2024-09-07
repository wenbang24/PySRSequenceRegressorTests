import numpy as np
from pysr import PySRSequenceRegressor

# Generate some data
X = [[1, 1, 1], [1, 1, 1]]
for i in range(2, 20):
    X.append([
        X[i-1][1] + X[i-2][0],
        X[i-1][0] + X[i-1][2],
        X[i-2][2] + X[i-2][1]
    ])
X = np.array(X)
model = PySRSequenceRegressor(
    recursive_history_length=2,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin", "exp"],
    niterations=100
)
print(X)
model.fit(X)
print(model.equations_)

# interpolation
pred = model.predict(X, num_predictions=19)[:-1]
print("Interpolation MSE: ", np.mean((pred - X[2:]) ** 2))
print("Interpolation MPE: ", np.mean((np.abs(pred - X[2:]))/X[2:]) * 100)

# extrapolation
true = X[-2:].tolist()
for i in range(18):
    true.append([
        true[-1][1] + true[-2][0],
        true[-1][0] + true[-1][2],
        true[-2][2] + true[-2][1]
    ])
true = np.asarray(true)[2:]
pred = model.predict(X, num_predictions=36)[18:]
print("Extrapolation MSE: ", np.mean((pred-true) ** 2))
print("Extrapolation MPE: ", np.mean((np.abs(pred - true))/true) * 100)