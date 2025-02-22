import pandas as pd

# Define file paths
data_path = "../data/smart_meters_london_2013_transposed.csv"
output_path = "../data/smart_meters_london_2013_transposed_cleaned.csv"

# Load dataset
def drop_first_column(path, output_path):
    data = pd.read_csv(path, low_memory=False)
    data = data.iloc[:, 1:]  # Drop the first column
    data.to_csv(output_path, index=False)
    print(f"Processed dataset saved at: {output_path}")

# Run script
drop_first_column(data_path, output_path)
