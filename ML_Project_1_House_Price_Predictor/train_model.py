
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Step 2: Load the dataset
df = pd.read_csv("housing.csv") 
df.head()


# Check the shape of the dataset
print("Shape:", df.shape)

# Check datatypes and null values
print("\nInfo:")
print(df.info())

# Summary statistics
print("\nDescribe:")
print(df.describe())

# Check for missing values visually
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu")
plt.title("Missing Value Heatmap")
plt.show()


# Plotting distribution of the target variable
plt.figure(figsize=(8,5))
sns.histplot(df['median_house_value'], bins=40, kde=True, color='orange')
plt.title('Distribution of House Prices')
plt.xlabel('Median House Value')
plt.ylabel('Count')
plt.show()



# Encode categorical column before correlation
df_encoded = pd.get_dummies(df, drop_first=True)


# Now plot the correlation heatmap
plt.figure(figsize=(12,8))
corr = df_encoded.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()



# Check total missing values
df.isnull().sum()



# Example: Fill missing numerical values with median
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# Check for categorical columns
df.select_dtypes(include=['object']).columns

# Encode 'ocean_proximity' using one-hot encoding
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)


from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


df_encoded.isnull().sum()



# Fill NaN values with median of each column
df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))

# ✅ Double check again
print("Still missing values?")
print(df_encoded.isnull().sum().sum())  # should be 0



# Split & train again
X = df_encoded.drop("median_house_value", axis=1)
y = df_encoded["median_house_value"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



assert not pd.isnull(X).values.any(), "X still has missing values!"


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Mean Absolute Error: {mae:.2f}")
print(f"✅ Mean Squared Error: {mse:.2f}")
print(f"✅ R² Score: {r2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.3, color='darkgreen')
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.savefig("accuracy.png")  # 📸 Save screenshot if needed
plt.show()


#this is the total evaluation of the linear regression model
✅ Mean Absolute Error: 50670.74
✅ Mean Squared Error: 4908476721.16
✅ R² Score: 0.6254





#now i will use random forest for the improvement


# Feature Engineering - Create new informative features
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



print("Random Forest Evaluation:")
print(f"✅ MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"✅ MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"✅ R² Score: {r2_score(y_test, y_pred):.4f}")


Random Forest Evaluation:
✅ MAE: 31492.94
✅ MSE: 2386213003.16
✅ R² Score: 0.8179


# To check acutual values vs predicted values
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.3, color='darkgreen')
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.savefig("accuracy.png")  
plt.show()



