import numpy as np
from pysr import PySRRegressor, PySRSequenceRegressor

# Generate some data
X = [1, 1]
for i in range(2, 20):
    X.append(X[i-1] + X[i-2])
X = np.array(X).reshape(-1, 1)
model = PySRSequenceRegressor(
    recursive_history_length=2,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin", "exp"],
    niterations=5, # 100 really isn't needed here
)
model.fit(X)
print(model.equations_)

print(model.get_best())

# interpolation
pred = model.predict(X, num_predictions=19)[:-1]
print("Interpolation MSE: ", np.mean((pred - X[2:].flatten()) ** 2))
print("Interpolation MPE: ", np.mean((np.abs(pred - X[2:].flatten()))/X[2:].flatten()) * 100)

# extrapolation
true = X[-2:].flatten().tolist()
for i in range(18):
    true.append(true[-1] + true[-2])
true = np.asarray(true)
pred = model.predict(X, num_predictions=36)[16:].flatten()
print("Extrapolation MSE: ", np.mean((pred-true) ** 2))
print("Extrapolation MPE: ", np.mean((np.abs(pred - true))/true) * 100)