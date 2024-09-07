import numpy as np
from pysr import PySRSequenceRegressor

# Generate some data
X = np.random.randn(20, 3) * 100
model = PySRSequenceRegressor(
    recursive_history_length=3,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin", "exp"],
    niterations=100
)
print(X)
model.fit(X)
print(model.equations_)

# interpolation
pred = model.predict(X, num_predictions=18)[:-1]
print("Interpolation MSE: ", np.mean((pred - X[3:]) ** 2))
print("Interpolation MPE: ", np.mean(np.abs((np.abs(pred - X[3:]))/X[3:])) * 100)

# extrapolation
true = np.random.randn(18, 3) * 100
pred = model.predict(X, num_predictions=35)[17:]
print("Extrapolation MSE: ", np.mean((pred-true) ** 2))
print("Extrapolation MPE: ", np.mean(np.abs((np.abs(pred - true))/true)) * 100)