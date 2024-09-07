import numpy as np
from pysr import PySRSequenceRegressor

# Generate some data
X = [[1, 1], [1, 1]]
for i in range(1, 20):
    X.append([
        np.exp(X[i-1][0] * 0.2) + X[i-2][1],
        X[i-1][0]
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
pred = model.predict(X, num_predictions=20)[:-1]
print("Interpolation MSE: ", np.mean((pred - X[2:]) ** 2))
print("Interpolation MPE: ", np.mean((np.abs(pred - X[2:]))/X[2:]) * 100)