import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import shutil

# Define file paths
cluster_data_dir = "../data/monthly_clusters"
week_cluster_data_dir = "../data/weekly_clusters"
plot_output_dir = "../results/plots/weekly_clusters"
if os.path.exists(plot_output_dir):
    shutil.rmtree(plot_output_dir)
os.makedirs(plot_output_dir, exist_ok=True)
os.makedirs(week_cluster_data_dir, exist_ok=True)


# Function to load a specific month cluster dataset
def load_month_cluster(cluster_id):
    cluster_file_path = os.path.join(cluster_data_dir,
                                     f"cluster_{cluster_id}.pkl")
    if os.path.exists(cluster_file_path):
        return pd.read_pickle(cluster_file_path)
    else:
        print(f"Cluster file {cluster_file_path} not found.")
        return None


# Function to aggregate data at the weekly level
def aggregate_weekly(data, timestamps):
    time_mappings = pd.Series(timestamps).dt.to_period("W").astype(str)
    aggregated_data = data.groupby(time_mappings, axis=1).mean()
    return aggregated_data


# Function to perform clustering on weekly aggregated data
def cluster_weekly_data(data, num_clusters):
    data = data.dropna()
    if data.shape[0] == 0:
        return np.array([])
    kmeans = KMeans(n_clusters=min(num_clusters, data.shape[0]),
                    random_state=42, n_init=10)
    return kmeans.fit_predict(data)


# Function to compute mean and std for weekly clusters
def compute_weekly_cluster_stats(data, clusters, cluster_id):
    data["weekly_cluster"] = clusters
    cluster_stats = {}

    for weekly_cluster_id in range(np.max(clusters) + 1):
        weekly_cluster_data = data[
            data["weekly_cluster"] == weekly_cluster_id].drop(
            columns=["weekly_cluster"])
        cluster_means = weekly_cluster_data.mean(axis=0)
        cluster_stds = weekly_cluster_data.std(axis=0)
        cluster_stats[weekly_cluster_id] = (
        cluster_means, cluster_stds, len(weekly_cluster_data))

        # Save weekly sub-cluster data
        weekly_cluster_file = os.path.join(week_cluster_data_dir,
                                           f"month_{cluster_id}_weekly_{weekly_cluster_id}.pkl")
        weekly_cluster_data.to_pickle(weekly_cluster_file)
        print(
            f"Saved weekly cluster {weekly_cluster_id} for month cluster {cluster_id} to {weekly_cluster_file}")

    return cluster_stats


# Function to plot weekly cluster statistics
def plot_weekly_cluster_stats(cluster_stats, cluster_id):
    for weekly_cluster_id, (means, stds, count) in cluster_stats.items():
        plt.figure(figsize=(10, 5))
        yerr = stds.values if not np.all(np.isnan(stds.values)) else None
        plt.errorbar(means.index, means.values, yerr=yerr, fmt='-o', capsize=5)
        plt.title(
            f"Monthly Cluster {cluster_id} - Weekly Cluster {weekly_cluster_id} - Mean and Std Dev\n({count} instances)")
        plt.xlabel("Week Interval")
        plt.ylabel("Mean kWh")
        plt.xticks(rotation=45)
        plt.grid()

        plot_path = os.path.join(plot_output_dir,
                                 f"month_{cluster_id}_weekly_{weekly_cluster_id}_stats.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    print(f"Weekly cluster plots saved in {plot_output_dir}")


# Define which monthly clusters to process
month_clusters_to_process = [
    0,
    2,
    4
]  # Specify the month cluster indices

weekly_k_values = {
    0: 4,
    2: 5,
    4: 3
}  # Specify k-values for KMeans per month cluster

# Process each selected monthly cluster
timestamps = pd.date_range(start="2013-01-01", periods=8760, freq="H")

for month_cluster_id in month_clusters_to_process:
    month_data = load_month_cluster(month_cluster_id)
    if month_data is not None:
        weekly_data = aggregate_weekly(
            month_data,
            timestamps
        )

        weekly_clusters = cluster_weekly_data(
            weekly_data,
            weekly_k_values[month_cluster_id]
        )

        weekly_cluster_stats = compute_weekly_cluster_stats(
            weekly_data,
            weekly_clusters,
            month_cluster_id
        )

        plot_weekly_cluster_stats(
            weekly_cluster_stats,
            month_cluster_id
        )