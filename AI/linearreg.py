import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
data = pd.read_csv(r"C:\Users\shevaun\Downloads\data.csv")

# Check data summary
print(data.describe())

# Extract SIZE (feature) and PRICE (target)
x = data['SIZE'].values
y = data['PRICE'].values

# Normalize the data (scaling values between 0 and 1)
x = (x - x.min()) / (x.max() - x.min())
y = (y - y.min()) / (y.max() - y.min())

# Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * x + c
    dm = (-2 / N) * np.sum(x * (y - y_pred))
    dc = (-2 / N) * np.sum(y - y_pred)
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Initial random values for m (slope) and c (intercept)
m, c = np.random.rand(), np.random.rand()
learning_rate = 0.01
epochs = 10  # Increased epochs for better convergence

# Training the model
for epoch in range(epochs):
    y_pred = m * x + c
    mse = mean_squared_error(y, y_pred)
    print(f"Epoch {epoch + 1}, MSE: {mse}, m: {m}, c: {c}")  # Print m and c
    m, c = gradient_descent(x, y, m, c, learning_rate)

# Plotting the final line of best fit
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m * x + c, color='red', label='Line of Best Fit')
plt.xlabel('Normalized Office Size')
plt.ylabel('Normalized Office Price')
plt.legend()
plt.show()

# Predicting the office price for size = 100 sq. ft
size = (100 - data['SIZE'].min()) / (data['SIZE'].max() - data['SIZE'].min())  # Normalized size
predicted_price = m * size + c
predicted_price_original_scale = predicted_price * (data['PRICE'].max() - data['PRICE'].min()) + data['PRICE'].min()
print(f"Predicted office price for size 100 sq. ft: {predicted_price_original_scale}")