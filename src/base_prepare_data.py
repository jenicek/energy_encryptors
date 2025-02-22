import pandas as pd
import numpy as np

# Define file paths
data_path = "../data/smart_meters_london_2013_transposed_100.csv"
output_pickle_path = "../data/smart_meters_cleaned.pkl"
output_numpy_path = "../data/smart_meters_cleaned.npy"

# Load CSV with proper settings
data = pd.read_csv(data_path, low_memory=False, header=None)

# Set the first row as column names and drop it from the dataframe
data.columns = data.iloc[0]
data = data[1:].reset_index(drop=True)

# Set the first column as the index for Pandas DataFrame
data.set_index(data.columns[0], inplace=True)

# Drop the first row in Pandas DataFrame
data = data.iloc[1:]

# Convert all columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Convert dataframe to a NumPy array (dropping the index column and first row)
numpy_data = data.to_numpy()
numpy_data = numpy_data[1:, :]

# Save dataframe as pickle (fastest for Pandas)
data.to_pickle(output_pickle_path)

# Save NumPy array
np.save(output_numpy_path, numpy_data)

print(f"Data saved as Pickle: {output_pickle_path}")
print(f"Data saved as NumPy array: {output_numpy_path}")
