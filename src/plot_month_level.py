import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths
data_path = "../data/smart_meters_london_2013_transposed.csv"
output_dir = "../results/plots"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data = pd.read_csv(data_path, low_memory=False, header=None)

# Drop the first column and first row
data = data.iloc[1:, 1:].reset_index(drop=True)

# Set first row as column names
data.columns = pd.to_datetime(data.iloc[0])
data = data[1:].reset_index(drop=True)

# Convert all columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Randomly select 20 rows
sampled_rows = data.sample(n=20, random_state=42)

# Convert column names (timestamps) to a DataFrame index for grouping
timestamps = pd.to_datetime(data.columns)
months = timestamps.month
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
               "Oct", "Nov", "Dec"]

# Compute monthly means for each sampled row and plot
for i, row in enumerate(sampled_rows.iterrows()):
    index, row_values = row
    monthly_means = pd.DataFrame(
        {'Month': months, 'Mean Consumption': row_values.values})
    monthly_means = monthly_means.groupby('Month').mean()
    monthly_means.index = [month_order[m - 1] for m in
                           monthly_means.index]  # Ensure correct month order
    monthly_means = monthly_means.reindex(
        month_order)  # Maintain chronological order

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(monthly_means.index, monthly_means['Mean Consumption'], marker='o',
             linestyle='-')
    plt.title(f"Monthly Mean Consumption for Row {index}")
    plt.xlabel("Month")
    plt.ylabel("Mean kWh")
    plt.xticks(rotation=45)
    plt.grid()

    # Save plot
    plot_path = os.path.join(output_dir, f"row_{index}_monthly_mean.png")
    plt.savefig(plot_path)
    plt.close()

print(f"Plots saved in {output_dir}")