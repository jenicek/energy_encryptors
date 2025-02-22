import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import shutil

# Define file paths
data_path = "../data/smart_meters_london_2013_transposed_100.csv"
output_synthetic_path = "../data/synthetic_smart_meters.pkl"
plot_output_dir = "../results/plots/monthly_clusters"
if os.path.exists(plot_output_dir):
    shutil.rmtree(plot_output_dir)
os.makedirs(plot_output_dir, exist_ok=True)

cluster_data_dir = "../data/monthly_clusters"
os.makedirs(plot_output_dir, exist_ok=True)
os.makedirs(cluster_data_dir, exist_ok=True)


# Load dataset
def load_data(path):
    data = pd.read_csv(path, low_memory=False, header=None)
    timestamps = pd.to_datetime(data.iloc[0, 0:])  # Extract column names
    data = data.iloc[1:, ].reset_index(
        drop=True)  # Drop the first row (date time)
    data = data.apply(pd.to_numeric, errors='coerce')
    return data, timestamps  # Return data and timestamps


# Function to aggregate data at a given time level
def aggregate_time_level(
        data,
        timestamps,
        time_level
):
    time_mappings = pd.Series(timestamps).dt.to_period(time_level).astype(str)
    aggregated_data = data.groupby(time_mappings, axis=1).mean()
    return aggregated_data


# Perform clustering
def cluster_data(
        data,
        num_clusters
):
    data = data.dropna()  # Drop rows with NaN before clustering
    if data.shape[0] == 0:
        return np.array([])  # Return empty array if no valid data remains
    kmeans = KMeans(n_clusters=min(num_clusters, data.shape[0]),
                    random_state=42, n_init=10)
    return kmeans.fit_predict(data)


# Compute mean and std per month for each cluster
def compute_cluster_stats(original_data, aggregated_data, clusters):
    aggregated_data["cluster"] = clusters
    original_data["cluster"] = clusters

    cluster_stats = {}
    for cluster_id in range(np.max(clusters) + 1):
        cluster_data_agg = aggregated_data[
            aggregated_data["cluster"] == cluster_id].drop(columns=["cluster"])
        cluster_data_full = original_data[
            original_data["cluster"] == cluster_id].drop(columns=["cluster"])
        cluster_means = cluster_data_agg.mean(axis=0)
        cluster_stds = cluster_data_agg.std(axis=0)
        cluster_size = len(cluster_data_agg)
        cluster_stats[cluster_id] = (cluster_means, cluster_stds, cluster_size)

        # Save the full cluster data (8760 timepoints)
        cluster_file_path = os.path.join(cluster_data_dir,
                                         f"cluster_{cluster_id}.pkl")
        cluster_data_full.to_pickle(cluster_file_path)
        print(f"Saved cluster {cluster_id} full data to {cluster_file_path}")

    return cluster_stats


# Generate plots for cluster statistics
def plot_cluster_stats(cluster_stats):
    for cluster_id, (means, stds, cluster_size) in cluster_stats.items():
        plt.figure(figsize=(10, 5))
        yerr = stds.values if not np.all(
            np.isnan(stds.values)) else None  # Handle missing std dev
        plt.errorbar(means.index, means.values, yerr=yerr, fmt='-o', capsize=5)
        plt.title(
            f"Cluster {cluster_id} - Mean and Std Dev\n({cluster_size} instances)")
        plt.xlabel("Time Interval")
        plt.ylabel("Mean kWh")
        plt.xticks(rotation=45)
        plt.grid()

        plot_path = os.path.join(plot_output_dir,
                                 f"cluster_{cluster_id}_stats.png")
        plt.savefig(plot_path, bbox_inches='tight')  # Prevent flashing
        plt.close()
    print(f"Cluster plots saved in {plot_output_dir}")


# Load data and run clustering
data, timestamps = load_data(data_path)
time_level = "M"  # Aggregation level: 'M' for month, 'W' for week, 'D' for day
aggregated_data = aggregate_time_level(data, timestamps, time_level)
clusters = cluster_data(aggregated_data, num_clusters=6)

# Compute and plot cluster statistics
cluster_stats = compute_cluster_stats(data, aggregated_data, clusters)
plot_cluster_stats(cluster_stats)