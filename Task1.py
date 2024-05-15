# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
url = "http://bit.ly/w-data"
data = pd.read_csv(url)

# Prepare the data
X = data.iloc[:, :-1].values  # Study hours (input)
y = data.iloc[:, 1].values    # Percentage (output)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the percentage for 9.25 hrs/day
hours = [[9.25]]
predicted_score = model.predict(hours)
print("Predicted score if a student studies for 9.25 hrs/day:", predicted_score[0])
