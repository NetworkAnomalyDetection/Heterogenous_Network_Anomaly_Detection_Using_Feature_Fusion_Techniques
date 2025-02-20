#This file is useful for extracting top 10 features which have greater importance in determining the probability of the transaction being anamolous
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Start timer
start_time = time.time()

# Load dataset
data = pd.read_csv('all_data.csv')

# Reduce dataset size for faster testing
data = data.sample(frac=0.1, random_state=42)

# Handle categorical target variable
if data['Label'].dtype == 'object':
    le = LabelEncoder()
    data['Label'] = le.fit_transform(data['Label'])

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Optimize data types
data = data.astype('float32')

# Separate features and target variable
X = data.drop(columns=['Label'])
y = data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor with optimizations
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Get feature importance
feature_importances = rf.feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'])
plt.title('Feature Importance using Random Forest Regressor')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Display top important features
print("Top 10 Important Features:")
print(feature_importance_df.head(10))

# Print execution time
print(f"Execution Time: {time.time() - start_time:.2f} seconds")
