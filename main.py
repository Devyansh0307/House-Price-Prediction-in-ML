# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: Make sure 'train.csv' and 'test.csv' are in the same directory.")
    exit()

# Keep a copy of the test IDs for the final submission
test_ids = test_df['Id']
# Drop the 'Id' column as it's not a predictive feature
train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

print("Data loaded successfully.")
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# --- Target Variable Analysis ---
# Plot a histogram to see the distribution of SalePrice
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.show()
# We can see the SalePrice is right-skewed. A log transformation can help normalize it.

# Apply log transformation to SalePrice to handle skewness
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

# Plot the transformed SalePrice
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Distribution of Log-Transformed SalePrice')
plt.show()


# --- Feature Correlation Analysis ---
# Select only numerical columns for correlation calculation to avoid errors
numerical_cols = train_df.select_dtypes(include=np.number).columns
correlation_matrix = train_df[numerical_cols].corr()

# Plot a heatmap of the correlations
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Find the top 10 features most correlated with SalePrice
print("Top 10 features correlated with SalePrice:")
print(correlation_matrix['SalePrice'].sort_values(ascending=False).head(11))

# Store the target variable and drop it from the training data
y_train = train_df['SalePrice']
train_df = train_df.drop('SalePrice', axis=1)

# Combine train and test data for consistent preprocessing
all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
print(f"Combined data shape: {all_data.shape}")

# --- Handle Missing Values ---
# Identify numerical and categorical columns
numerical_cols = all_data.select_dtypes(include=np.number).columns
categorical_cols = all_data.select_dtypes(include=['object']).columns

# Fill missing numerical values with the median of the column
for col in numerical_cols:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# Fill missing categorical values with the mode (most frequent value)
for col in categorical_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# --- One-Hot Encode Categorical Features ---
# This converts categories into a numerical format the model can understand
all_data = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)

print(f"Shape after one-hot encoding: {all_data.shape}")

# --- Separate back into training and test sets ---
# Find the split point
split_point = len(train_df)
X_train = all_data[:split_point]
X_test = all_data[split_point:]

print(f"Final training data shape: {X_train.shape}")
print(f"Final test data shape: {X_test.shape}")

# Split the training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_split, y_train_split)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"\nModel Validation RMSE: {rmse}")

# Visualize the predictions vs actual values
plt.figure(figsize=(8, 8))
plt.scatter(y_val, y_pred_val)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', lw=2) # a perfect prediction line
plt.xlabel('Actual (Log-Transformed) Prices')
plt.ylabel('Predicted (Log-Transformed) Prices')
plt.title('Validation: Actual vs. Predicted Prices')
plt.show()

# Retrain the model on the full training data for best performance
model.fit(X_train, y_train)

# Make predictions on the final test set
final_predictions = model.predict(X_test)

# The model predicted the log of the price, so we need to convert it back
final_predictions = np.expm1(final_predictions)

# Create the submission file
submission_df = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_predictions
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("First 5 predictions:")
print(submission_df.head())