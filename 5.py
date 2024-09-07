import numpy as np
from pysr import PySRSequenceRegressor

# Generate some data
X = [[1, 1]]
for i in range(1, 20):
    X.append([
        X[i-1][1] ** 1.1,
        np.cos(X[i-1][0])
    ])
X = np.array(X)
model = PySRSequenceRegressor(
    recursive_history_length=1,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin", "exp"],
    niterations=100
)
print(X)
model.fit(X)
print(model.equations_)

# interpolation
pred = model.predict(X, num_predictions=20)[:-1]
print("Interpolation MSE: ", np.mean((pred - X[1:]) ** 2))
print("Interpolation MPE: ", np.mean((np.abs(pred - X[1:]))/X[1:]) * 100)

# extrapolation
true = X[-1:].tolist()
for i in range(18):
    true.append([
        true[-1][1] / 3.1324,
        true[-1][0] * 1.1291 + 4.3244
    ])
true = np.asarray(true)
pred = model.predict(X, num_predictions=37)[18:]
print("Extrapolation MSE: ", np.mean((pred-true) ** 2))
print("Extrapolation MPE: ", np.mean((np.abs(pred - true))/true) * 100)