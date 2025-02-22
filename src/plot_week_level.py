import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths
data_path = "../data/smart_meters_london_2013_transposed.csv"
output_dir = "../results/plots/week_level_plots"
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
weeks = timestamps.isocalendar().week
week_labels = [f"Week {w}" for w in range(1, 53)]  # Generate week labels

# Compute weekly means for each sampled row and plot
for i, row in enumerate(sampled_rows.iterrows()):
    index, row_values = row
    weekly_means = pd.DataFrame(
        {'Week': weeks, 'Mean Consumption': row_values.values})
    weekly_means = weekly_means.groupby('Week').mean()
    weekly_means.index = [f"Week {w}" for w in weekly_means.index]
    weekly_means = weekly_means.reindex(
        week_labels)  # Ensure correct chronological order

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(weekly_means.index, weekly_means['Mean Consumption'], marker='o',
             linestyle='-')
    plt.title(f"Weekly Mean Consumption for household {index}")
    plt.xlabel("Week")
    plt.ylabel("Mean kWh")
    plt.xticks(rotation=45, fontsize=8)
    plt.grid()

    # Save plot
    plot_path = os.path.join(output_dir, f"row_{index}_weekly_mean.png")
    plt.savefig(plot_path)
    plt.close()

print(f"Plots saved in {output_dir}")
