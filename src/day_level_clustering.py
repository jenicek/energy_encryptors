import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import shutil

# Define file paths
week_cluster_data_dir = "../data/weekly_clusters"
day_cluster_data_dir = "../data/daily_clusters"
plot_output_dir = "../results/plots/daily_clusters"
if os.path.exists(plot_output_dir):
    shutil.rmtree(plot_output_dir)
os.makedirs(plot_output_dir, exist_ok=True)
os.makedirs(day_cluster_data_dir, exist_ok=True)

# Function to load a specific weekly cluster dataset
def load_week_cluster(month_id, week_id):
    cluster_file_path = os.path.join(week_cluster_data_dir,
                                     f"month_{month_id}_weekly_{week_id}.pkl")
    if os.path.exists(cluster_file_path):
        return pd.read_pickle(cluster_file_path)
    else:
        print(f"Cluster file {cluster_file_path} not found.")
        return None

# Function to aggregate data at the daily level
def aggregate_daily(data, timestamps):
    time_mappings = pd.Series(timestamps).dt.to_period("D").astype(str)
    aggregated_data = data.groupby(time_mappings, axis=1).mean()
    return aggregated_data

# Function to perform clustering on daily aggregated data
def cluster_daily_data(data, num_clusters):
    data = data.dropna()
    if data.shape[0] == 0:
        return np.array([])
    kmeans = KMeans(n_clusters=min(num_clusters, data.shape[0]),
                    random_state=42, n_init=10)
    return kmeans.fit_predict(data)

# Function to compute mean and std for daily clusters
def compute_daily_cluster_stats(
        data,
        clusters,
        month_id,
        week_id
):
    data["daily_cluster"] = clusters
    cluster_stats = {}

    for daily_cluster_id in range(np.max(clusters) + 1):
        daily_cluster_data = data[
            data["daily_cluster"] == daily_cluster_id].drop(
            columns=["daily_cluster"])
        cluster_means = daily_cluster_data.mean(axis=0)
        cluster_stds = daily_cluster_data.std(axis=0)
        cluster_stats[daily_cluster_id] = (
        cluster_means, cluster_stds, len(daily_cluster_data))

        # Save daily sub-cluster data
        daily_cluster_file = os.path.join(day_cluster_data_dir,
                                           f"month_{month_id}_weekly_{week_id}_daily_{daily_cluster_id}.pkl")
        daily_cluster_data.to_pickle(daily_cluster_file)
        print(
            f"Saved daily cluster {daily_cluster_id} for week {week_id} in month {month_id} to {daily_cluster_file}")

    return cluster_stats

# Function to plot daily cluster statistics
def plot_daily_cluster_stats(cluster_stats, month_id, week_id):
    for daily_cluster_id, (means, stds, count) in cluster_stats.items():
        plt.figure(figsize=(10, 5))
        yerr = stds.values if not np.all(np.isnan(stds.values)) else None
        plt.errorbar(means.index, means.values, yerr=yerr, fmt='-o', capsize=5)
        plt.title(
            f"Month {month_id} - Week {week_id} - Daily Cluster {daily_cluster_id} - Mean and Std Dev\n({count} instances)")
        plt.xlabel("Day Interval")
        plt.ylabel("Mean kWh")
        plt.xticks(rotation=45)
        plt.grid()

        plot_path = os.path.join(plot_output_dir,
                                 f"month_{month_id}_weekly_{week_id}_daily_{daily_cluster_id}_stats.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    print(f"Daily cluster plots saved in {plot_output_dir}")

# Define which weekly clusters to process
week_clusters_to_process = {
    0: [0, 2, 3, 4, 7],
    2: [0, 1, 2],
    3: [0, 1, 2],
    5: [0, 1, 4],
}  # Specify which week clusters to process per month

daily_k_values = {
    (0, 0): 3,
    (0, 2): 3,
    (0, 3): 3,
    (0, 4): 3,
    (0, 7): 3,
    (2, 0): 3,
    (2, 1): 3,
    (2, 2): 3,
    (3, 0): 3,
    (3, 1): 3,
    (3, 2): 3,
    (5, 0): 3,
    (5, 1): 3,
    (5, 4): 3,
}  # Specify k-values for KMeans per week cluster

# Process each selected weekly cluster
timestamps = pd.date_range(start="2013-01-01", periods=8760, freq="H")

for month_id, week_ids in week_clusters_to_process.items():
    for week_id in week_ids:
        week_data = load_week_cluster(month_id, week_id)
        if week_data is not None:
            daily_data = aggregate_daily(
                week_data,
                timestamps
            )

            daily_clusters = cluster_daily_data(
                daily_data,
                daily_k_values[(month_id, week_id)]
            )

            daily_cluster_stats = compute_daily_cluster_stats(
                week_data,
                daily_clusters,
                month_id,
                week_id
            )

            plot_daily_cluster_stats(
                daily_cluster_stats,
                month_id,
                week_id
            )
