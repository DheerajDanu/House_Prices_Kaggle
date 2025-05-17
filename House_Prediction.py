import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv(r'D:\PYTHON_SCRIPT\Kaggle_House_Prices\train.csv')
test_data = pd.read_csv(r'D:\PYTHON_SCRIPT\Kaggle_House_Prices\test.csv')

# Log-transform target to handle outliers
y = np.log1p(train_data['SalePrice'])  # log1p = log(1 + x) to avoid log(0)

# Select features
x = train_data.drop(['Id', 'SalePrice'], axis=1)

# Identify numerical and categorical features
numerical_features = x.select_dtypes(include=['int64', 'float64']).columns
categorical_features = x.select_dtypes(include=['object']).columns

# Handle missing values
x[numerical_features] = x[numerical_features].fillna(x[numerical_features].median())
x[categorical_features] = x[categorical_features].fillna(x[categorical_features].mode().iloc[0])

# Split data before encoding
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Apply target encoding
te = TargetEncoder(cols=categorical_features)
x_train[categorical_features] = te.fit_transform(x_train[categorical_features], y_train)
x_val[categorical_features] = te.transform(x_val[categorical_features])

# Train first model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate (predict in log scale, convert back for MSE)
y_pred = model.predict(x_val)
mse = mean_squared_error(np.expm1(y_val), np.expm1(y_pred))  # expm1 = exp(x) - 1
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize
feature_importance.head(10).plot(kind='bar', title='Top 10 Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# Train second model with top 10 features
top_10 = feature_importance.head(10).index
model_2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_2.fit(x_train[top_10], y_train)
y_pred2 = model_2.predict(x_val[top_10])

# Evaluate second model
mse_top = mean_squared_error(np.expm1(y_val), np.expm1(y_pred2))
rmse_top = np.sqrt(mse_top)
print(f'Mean Squared Error (Top 10 Features): {mse_top}')
print(f'Root Mean Squared Error (Top 10 Features): {rmse_top}')

# Prepare test set for submission
X_test = test_data.drop(['Id'], axis=1)
X_test[numerical_features] = X_test[numerical_features].fillna(X_test[numerical_features].median())
X_test[categorical_features] = X_test[categorical_features].fillna(X_test[categorical_features].mode().iloc[0])
X_test[categorical_features] = te.transform(X_test[categorical_features])
test_preds = np.expm1(model.predict(X_test))  # Convert back from log scale
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_preds})
submission.to_csv(r'D:\PYTHON_SCRIPT\Kaggle_House_Prices\submission.csv', index=False)
print("Submission file created: submission.csv")