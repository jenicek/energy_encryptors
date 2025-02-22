import pandas as pd

# Define file paths
data_path = "../data/smart_meters_london_2013.csv"
synthetic_path = "../data/synthetic_smart_meters.pkl"
output_path = "../data/smart_meters_london_2013_synthetic.csv"

# Load the real dataset
real_data = pd.read_csv(data_path, low_memory=False)

# Load the synthetic dataset
synthetic_data = pd.read_pickle(synthetic_path)

# Transpose the synthetic dataset
synthetic_data = synthetic_data.T

# Add the first column of the real dataset as the first column in the synthetic dataset
synthetic_data.insert(0, real_data.columns[0], real_data.iloc[:, 0])

# Save the merged dataset
synthetic_data.to_csv(output_path, index=False)
print(f"Merged dataset saved to {output_path}")
