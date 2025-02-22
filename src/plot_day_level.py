import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Define file paths
data_path = "../data/smart_meters_london_2013_transposed.csv"
output_dir = "../results/plots/day_level_plots"
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
days = timestamps.strftime('%Y-%m-%d')  # Extract day in YYYY-MM-DD format
unique_days = sorted(set(days))  # Maintain correct chronological order

# Compute daily means for each sampled row and plot
for i, row in tqdm(enumerate(sampled_rows.iterrows()), total=len(sampled_rows),
                   desc="Generating Day-Level Plots"):
    index, row_values = row
    daily_means = pd.DataFrame(
        {'Day': days, 'Mean Consumption': row_values.values})
    daily_means = daily_means.groupby('Day').mean()
    daily_means = daily_means.reindex(
        unique_days)  # Ensure correct chronological order

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(daily_means.index, daily_means['Mean Consumption'], marker='o',
             linestyle='-')
    plt.title(f"Daily Mean Consumption for household {index}")
    plt.xlabel("Day")
    plt.ylabel("Mean kWh")
    plt.xticks(rotation=45, fontsize=6)
    plt.grid()

    # Save plot
    plot_path = os.path.join(output_dir, f"row_{index}_daily_mean.png")
    plt.savefig(plot_path)
    plt.close()

print(f"Plots saved in {output_dir}")
