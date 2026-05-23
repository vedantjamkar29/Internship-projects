# House Price Prediction Project
# Simple Machine Learning Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Sample house data
data = {
    'Area': [1000, 1200, 1500, 1800, 2000],
    'Bedrooms': [2, 2, 3, 3, 4],
    'Bathrooms': [1, 2, 2, 3, 3],
    'Price': [25, 30, 45, 50, 60]
}

# Create dataframe
df = pd.DataFrame(data)

# Input and output
X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[1600, 3, 2]])

print("Predicted House Price:", prediction[0], "Lakhs")

# Test model accuracy
y_pred = model.predict(X_test)

error = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", error)