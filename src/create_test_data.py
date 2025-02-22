import pandas as pd
import os

# Define file paths
data_path = "../data/smart_meters_london_2013_transposed.csv"
output_test_data_path = "../data/smart_meters_100_instances.csv"

# Load dataset
def load_data(path):
    data = pd.read_csv(path, low_memory=False, header=None)
    data = data.iloc[1:, 1:].reset_index(drop=True)
    timestamps = pd.to_datetime(data.iloc[0])
    data = data[1:].reset_index(drop=True)
    data.columns = timestamps
    data = data.apply(pd.to_numeric, errors='coerce')
    return data

# Save a smaller subset of the dataset for testing
def save_test_data(data, path, num_samples=100):
    test_data = data.sample(n=min(num_samples, data.shape[0]), random_state=42)
    test_data.to_csv(path, index=False)
    print(f"Test dataset with {num_samples} instances saved at: {path}")

# Run the script
data = load_data(data_path)
save_test_data(data, output_test_data_path)