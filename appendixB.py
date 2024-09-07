import numpy as np
from pysr import PySRRegressor

y = [1, 1]
for i in range(38):
    y.append(y[-1] + y[-2])
yfuture = np.asarray(y[20:]).reshape(-1, 1)
y = np.asarray(y[:20]).reshape(-1, 1)

X = np.arange(1, 21).reshape(-1, 1)
Xfuture = np.arange(21, 41).reshape(-1, 1)

model = PySRRegressor(
    binary_operators=["+", "*", "-", "/", "^"],
    unary_operators=["sin", "exp", "sqrt"],
    niterations=100,
)
model.fit(X, y)

# interpolation
pred = model.predict(X)
print("Interpolation MSE: ", np.mean((pred - y) ** 2))
print("Interpolation MPE: ", np.mean((np.abs(pred - y))/y) * 100)

# extrapolation
pred = model.predict(Xfuture)
print("Extrapolation MSE: ", np.mean((pred-yfuture) ** 2))
print("Extrapolation MPE: ", np.mean((np.abs(pred - yfuture))/yfuture) * 100)

print(model.latex())