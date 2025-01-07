import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib

# Load datasets
train_df = pd.read_csv("Train_Data_CSV.csv")
test_df = pd.read_csv("Test_Data_CSV.csv")

# Function to create sequences
def x_reshape(df, columns, sequence_length):
    data = df[columns].values
    num_elements = data.shape[0]
    for start, stop in zip(range(0, num_elements-sequence_length-10),  # Predict 10 timesteps ahead
                           range(sequence_length, num_elements-10)):
        yield data[start:stop, :]

# Function to extract input sequences
def get_x_slices(df, feature_columns):
    feature_list = [list(x_reshape(df[df['Data_No'] == i], feature_columns, 20))  # 20-step sequences
                    for i in range(1, df['Data_No'].nunique() + 1) if len(df[df['Data_No'] == i]) > 20]
    feature_array = np.concatenate(feature_list, axis=0).astype(np.float64)
    return feature_array

# Function to extract target values (RUL)
def y_reshape(df, sequence_length, target_column=['Differential_pressure']):
    data = df[target_column].values
    num_elements = data.shape[0]
    return data[sequence_length+10:num_elements, :]

def get_y_slices(df):
    label_list = [y_reshape(df[df['Data_No'] == i], 20, target_column=['Differential_pressure']) 
                  for i in range(1, df['Data_No'].nunique()+1)]
    label_array = np.concatenate(label_list).astype(np.float64)
    return label_array

# Feature selection
features = ['Differential_pressure']

# Extract sequences
X_train = get_x_slices(train_df, features)
X_test = get_x_slices(test_df, features)
y_train = get_y_slices(train_df)
y_test = get_y_slices(test_df)

# Ensure correct shapes
X_train = np.squeeze(X_train, axis=2)
y_train = np.squeeze(y_train, axis=1)
X_test = np.squeeze(X_test, axis=2)
y_test = np.squeeze(y_test, axis=1)

# Split train/validation
train_size = int(len(X_train) * 0.9)
X_train, X_val = X_train[:train_size], X_train[train_size:]
y_train, y_val = y_train[:train_size], y_train[train_size:]

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

# Evaluate the model
def evaluate(y_true, y_pred, label="test"):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} set - MAE: {mae:.4f}, R2: {r2:.4f}")

evaluate(y_train, y_train_pred, label="train")
evaluate(y_test, y_test_pred, label="test")

# Save model and scaler
joblib.dump(lin_reg, "linear_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Linear Regression model and scaler saved successfully.")


