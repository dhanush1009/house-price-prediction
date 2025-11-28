import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# Create a simple dummy model for testing
# This simulates a house price prediction model
X_train = np.array([
    [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23, -122.23],
    [8.3014, 21.0, 6.24, 0.97, 2401, 2.11, 37.88, -122.23, -122.23],
    [7.2574, 52.0, 8.29, 1.07, 496, 2.80, 37.85, -122.24, -122.24],
])

y_train = np.array([4.526, 3.585, 3.521])  # Sample prices

# Train a simple model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'house_model.pkl')
print("Model created and saved as house_model.pkl")
