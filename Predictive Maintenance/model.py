import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit

# Define dataset directories
dataset_base_path = "/Users/TonyMench/Desktop/Predictive Maintenance/ml-service"
dataset_paths = {
    "test1": os.path.join(dataset_base_path, "test1"),
    "test2": os.path.join(dataset_base_path, "test2"),
    "test3": os.path.join(dataset_base_path, "test3"),
}

# Define function to extract features from vibration data
def extract_features(dataset_path):
    features = ['mean', 'std', 'skew', 'kurtosis', 'entropy', 'rms', 'max', 'p2p', 'crest', 'clearance', 'shape', 'impulse']
    bearings = ['B1', 'B2', 'B3', 'B4']  # Four bearings in dataset
    
    # Create DataFrame to store features
    columns = [f"{b}_{f}" for b in bearings for f in features]
    df = pd.DataFrame(columns=columns)
    
    for filename in sorted(os.listdir(dataset_path)):
        file_path = os.path.join(dataset_path, filename)
        raw_data = pd.read_csv(file_path, sep='\t', header=None)

        # Compute statistical features
        mean_abs = raw_data.abs().mean().values
        std = raw_data.std().values
        skew = raw_data.skew().values
        kurtosis = raw_data.kurtosis().values
        entropy_vals = np.array([entropy(pd.cut(raw_data[col], 500).value_counts()) for col in raw_data.columns])
        rms = np.sqrt(np.mean(raw_data**2, axis=0))
        max_abs = raw_data.abs().max().values
        p2p = raw_data.max() - raw_data.min()
        crest = max_abs / rms
        clearance = ((np.sqrt(raw_data.abs()).sum() / len(raw_data))**2).values
        shape = rms / mean_abs
        impulse = max_abs / mean_abs

        # Combine features
        feature_data = np.concatenate([mean_abs, std, skew, kurtosis, entropy_vals, rms, max_abs, p2p, crest, clearance, shape, impulse])
        df.loc[filename] = feature_data

    return df

# Function to apply PCA for degradation trend analysis
def apply_pca(feature_df):
    pca = PCA(n_components=1)  # Reduce to 1 component
    health_indicator = pca.fit_transform(feature_df)
    
    feature_df["Health_Index"] = health_indicator
    print("Explained Variance:", pca.explained_variance_ratio_[0])
    
    return feature_df

# Function to fit exponential degradation model
def exponential_degradation(df, base=250):
    x = np.array(df.index)
    y = np.array(df["Health_Index"])
    
    def exp_model(x, a, b):
        return a * np.exp(b * x)

    fit = curve_fit(exp_model, x[-base:], y[-base:], p0=[0.01, 0.001])
    
    return fit

# Function to predict failure cycle
def predict_failure_cycle(fit):
    a, b = fit[0]
    failure_threshold = 2  # Define failure threshold
    predicted_fail_cycle = (np.log(failure_threshold / a)) / abs(b)
    return predicted_fail_cycle

# Process each test dataset
results = {}

for test_name, test_path in dataset_paths.items():
    print(f"Processing {test_name}...")

    # Step 1: Extract features
    feature_data = extract_features(test_path)

    # Step 2: Apply PCA
    feature_data = apply_pca(feature_data)

    # Step 3: Fit degradation model
    fit = exponential_degradation(feature_data)

    # Step 4: Predict failure cycle
    predicted_failure = predict_failure_cycle(fit)
    results[test_name] = predicted_failure

    print(f"Predicted Failure Cycle for {test_name}: {predicted_failure:.2f}")

# Display results
for test, failure_cycle in results.items():
    print(f"{test} → Predicted Failure Cycle: {failure_cycle:.2f}")

