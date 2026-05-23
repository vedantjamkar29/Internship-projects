# Student Performance Prediction System
# Made By: Vedant

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Student dataset
data = {
    'Study_Hours': [2, 4, 6, 8, 10],
    'Attendance': [60, 70, 80, 90, 95],
    'Previous_Marks': [50, 55, 65, 75, 85],
    'Final_Marks': [52, 60, 70, 82, 90]
}

# Create dataframe
df = pd.DataFrame(data)

# Input and output data
X = df[['Study_Hours', 'Attendance', 'Previous_Marks']]
y = df['Final_Marks']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict final marks
prediction = model.predict([[7, 85, 72]])

print("Predicted Final Marks:", prediction[0])

# Test model
y_pred = model.predict(X_test)

# Calculate error
error = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", error)