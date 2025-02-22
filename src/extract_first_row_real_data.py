import pandas as pd

# Define file paths
data_path = "../data/smart_meters_london_2013.csv"
first_column_pickle_path = "../data/first_column_smart_meters.pkl"

# Load the real dataset
real_data = pd.read_csv(data_path, low_memory=False)

# Extract the first column
first_column = real_data.iloc[:, 0]

# Save the first column as a pickle file
first_column.to_pickle(first_column_pickle_path)
print(f"First column extracted and saved to {first_column_pickle_path}")
