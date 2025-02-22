import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

pickle_path = "../data/smart_meters_london_2013_transposed_pandas.pkl"
output_dir = "../results/plots"

# Load dataset from pickle
data = pd.read_pickle(pickle_path)

# Basic information and statistics
df_desc = data.describe()
df_info = data.info()

# Define a function to save plots
def save_plot(fig, filename):
    fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close(fig)

# 1. Distribution of Energy Consumption
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data.iloc[:, 1:].values.flatten(), bins=50, kde=True, ax=ax)
ax.set_title("Distribution of Energy Consumption (kWh)")
ax.set_xlabel("kWh")
ax.set_ylabel("Frequency")
save_plot(fig, "energy_distribution.png")

# 2. Missing Values Heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.isnull(), cmap='viridis', cbar=False, ax=ax)
ax.set_title("Missing Values Heatmap")
save_plot(fig, "missing_values_heatmap.png")

# 3. Sample Time-Series Plot
sample_house = data.sample(1).iloc[:, 1:].T
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sample_house, marker='o', linestyle='-')
ax.set_title("Sample Time-Series Consumption for One Entity")
ax.set_xlabel("Timepoints")
ax.set_ylabel("kWh")
save_plot(fig, "sample_time_series.png")

# 4. Boxplot of Energy Consumption Over Time
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=data.iloc[:, 1:], ax=ax)
ax.set_title("Energy Consumption Over Time (Boxplot)")
ax.set_xlabel("Timepoints")
ax.set_ylabel("kWh")
save_plot(fig, "boxplot_over_time.png")

# 5. Correlation Heatmap
corr_matrix = data.iloc[:, 1:].corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt='.2f', ax=ax)
ax.set_title("Correlation Heatmap Between Timepoints")
save_plot(fig, "correlation_heatmap.png")

# 6. Average Energy Consumption Over Time
fig, ax = plt.subplots(figsize=(12, 6))
mean_consumption = data.iloc[:, 1:].mean()
ax.plot(mean_consumption, marker='o', linestyle='-', color='b')
ax.set_title("Average Energy Consumption Over Time")
ax.set_xlabel("Timepoints")
ax.set_ylabel("Mean kWh")
save_plot(fig, "mean_energy_over_time.png")

# 7. Variability in Consumption Over Time
fig, ax = plt.subplots(figsize=(12, 6))
std_dev = data.iloc[:, 1:].std()
ax.plot(std_dev, marker='o', linestyle='-', color='r')
ax.set_title("Variability in Energy Consumption Over Time")
ax.set_xlabel("Timepoints")
ax.set_ylabel("Standard Deviation of kWh")
save_plot(fig, "std_dev_energy_over_time.png")

# 8. Pairplot for a Small Sample
sample_df = data.sample(n=100, random_state=42).iloc[:, 1:6]
fig = sns.pairplot(sample_df)
fig.fig.suptitle("Pairplot of Sample Energy Consumption Data", y=1.02)
save_plot(fig.fig, "pairplot_sample.png")

# 9. Histogram of Mean Consumption Per Entity
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data.iloc[:, 1:].mean(axis=1), bins=50, kde=True, ax=ax)
ax.set_title("Histogram of Mean Energy Consumption per Entity")
ax.set_xlabel("Mean kWh")
ax.set_ylabel("Frequency")
save_plot(fig, "histogram_mean_consumption.png")

# 10. Timepoint Clustering via PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data.iloc[:, 1:].dropna())
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], alpha=0.5)
ax.set_title("PCA Projection of Timepoints")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
save_plot(fig, "pca_projection.png")

print(f"EDA plots saved in {output_dir}")
