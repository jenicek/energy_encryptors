import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

# Define file paths
data_path = "../data/smart_meters_london_2013_transposed.csv"
output_synthetic_path = "../data/synthetic_smart_meters.pkl"


# Load dataset
def load_data(path):
    data = pd.read_csv(path, low_memory=False, header=None)
    data = data.iloc[1:, 1:].reset_index(drop=True)
    timestamps = pd.to_datetime(data.iloc[0])
    data = data[1:].reset_index(drop=True)
    data.columns = timestamps
    data = data.apply(pd.to_numeric, errors='coerce')
    return data.T  # Transpose so time becomes the index


# Compute monthly means
def compute_monthly_means(data):
    data.index = pd.to_datetime(
        data.index)  # Ensure the index is a DatetimeIndex
    return data.resample("ME").mean().dropna(how='all').T  # Drop empty rows


# Compute weekly means
def compute_weekly_means(data):
    data.index = pd.to_datetime(data.index)
    return data.resample("W").mean().dropna(how='all').T


# Compute daily means
def compute_daily_means(data):
    data.index = pd.to_datetime(data.index)
    return data.resample("D").mean().dropna(how='all').T


# Perform clustering
def cluster_data(data, num_clusters):
    data = data.dropna()  # Drop rows with NaN before clustering
    if data.shape[0] == 0:
        return np.array([])  # Return empty array if no valid data remains
    kmeans = KMeans(n_clusters=min(num_clusters, data.shape[0]),
                    random_state=42, n_init=10)
    return kmeans.fit_predict(data)


# Generate synthetic samples using Gaussian noise
def generate_synthetic_samples(subgroup, num_samples):
    mean_pattern = subgroup.mean(axis=0)
    std_pattern = subgroup.std(axis=0)
    synthetic_data = np.array([
        mean_pattern + np.random.normal(0, std_pattern, size=mean_pattern.shape)
        for _ in range(num_samples)
    ])
    return synthetic_data


# Main function to create synthetic dataset
def create_synthetic_dataset(
        data,
        num_monthly_clusters=10,
        num_weekly_clusters=5,
        num_daily_clusters=5
):
    monthly_means = compute_monthly_means(data)
    monthly_clusters = cluster_data(monthly_means, num_monthly_clusters)

    data = data.iloc[:len(
        monthly_clusters)].copy()  # Ensure data size matches cluster assignment
    data["monthly_cluster"] = monthly_clusters

    synthetic_samples = []

    for month_cluster in tqdm(range(num_monthly_clusters),
                              desc="Processing Monthly Clusters"):
        monthly_subset = data[data["monthly_cluster"] == month_cluster].drop(
            columns=["monthly_cluster"])
        if monthly_subset.shape[0] == 0:
            continue

        weekly_means = compute_weekly_means(monthly_subset)
        weekly_clusters = cluster_data(weekly_means, num_weekly_clusters)
        if len(weekly_clusters) == 0:
            continue

        weekly_subset = monthly_subset.iloc[:len(weekly_clusters)].copy()
        weekly_subset["weekly_cluster"] = weekly_clusters

        for week_cluster in range(num_weekly_clusters):
            weekly_cluster_subset = weekly_subset[
                weekly_subset["weekly_cluster"] == week_cluster].drop(
                columns=["weekly_cluster"])
            if weekly_cluster_subset.shape[0] == 0:
                continue

            daily_means = compute_daily_means(weekly_cluster_subset)
            daily_clusters = cluster_data(daily_means, num_daily_clusters)
            if len(daily_clusters) == 0:
                continue

            daily_subset = weekly_cluster_subset.iloc[
                           :len(daily_clusters)].copy()
            daily_subset["daily_cluster"] = daily_clusters

            for day_cluster in range(num_daily_clusters):
                final_subset = daily_subset[
                    daily_subset["daily_cluster"] == day_cluster].drop(
                    columns=["daily_cluster"])
                if final_subset.shape[0] == 0:
                    continue

                num_samples = final_subset.shape[0]
                synthetic_data = generate_synthetic_samples(final_subset,
                                                            num_samples)
                synthetic_samples.append(synthetic_data)

    if len(synthetic_samples) == 0:
        print("No valid data found for synthetic generation.")
        return None

    synthetic_dataset = np.vstack(synthetic_samples)
    synthetic_df = pd.DataFrame(synthetic_dataset, columns=data.columns[
                                                           :-3])  # Drop cluster columns
    synthetic_df.to_pickle(output_synthetic_path)
    print(f"Synthetic dataset saved at: {output_synthetic_path}")
    return synthetic_df


# Load data and create synthetic dataset
data = load_data(data_path)
synthetic_data = create_synthetic_dataset(data)
