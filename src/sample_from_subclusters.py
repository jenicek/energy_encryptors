import pandas as pd
import numpy as np
import os
import random


# Function to get all file paths from a directory
def get_all_filepaths(directory):
    """
    Retrieve all .pkl file paths from a given directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing .pkl files.

    Returns
    -------
    list of str
        List of file paths to all .pkl files in the directory.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if
            f.endswith(".pkl")]


# Function to load cluster data from specified .pkl files
def load_cluster_data(filepaths):
    """
    Load cluster data from the specified list of .pkl file paths.

    Parameters
    ----------
    filepaths : list of str
        List of file paths pointing to .pkl files containing cluster data.

    Returns
    -------
    list of pandas.DataFrame
        A list of DataFrames, each representing a loaded cluster dataset.
    """
    cluster_data = []
    for filepath in filepaths:
        if os.path.exists(filepath):
            df = pd.read_pickle(filepath)
            cluster_data.append(df)
        else:
            print(f"Warning: File {filepath} not found.")
    return cluster_data


# Function to generate synthetic data from a cluster
def generate_synthetic_data(df, num_samples):
    """
    Generate synthetic instances from a given cluster dataset.

    The synthetic data is created by calculating the mean and standard deviation
    of each timepoint (column) and sampling new instances using a normal distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing real cluster data with instances as rows
        and timepoints as columns.
    num_samples : int
        The number of synthetic samples to generate.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the generated synthetic instances with the
        same number of columns (timepoints) as the original dataset.
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    synthetic_samples = np.random.normal(loc=means, scale=stds,
                                         size=(num_samples, df.shape[1]))
    synthetic_samples = np.clip(synthetic_samples, a_min=0,
                                a_max=None)  # Ensure no negative values
    return pd.DataFrame(synthetic_samples, columns=df.columns)


# Function to create synthetic dataset from multiple clusters
def create_synthetic_dataset(
        daily_dir,
        selected_weekly_files,
        selected_monthly_files
):
    """
    Construct a synthetic dataset by iterating over multiple cluster datasets.

    For each cluster dataset:
    - Compute the mean and standard deviation per timepoint.
    - Sample new instances based on these statistics.
    - Append the generated synthetic samples to a combined dataset.
    - Shuffle the final dataset before saving.

    Parameters
    ----------
    daily_dir : str
        Path to the directory containing all daily cluster .pkl files.
    selected_weekly_files : list of str
        List of selected .pkl file paths from weekly clusters.
    selected_monthly_files : list of str
        List of selected .pkl file paths from monthly clusters.

    Returns
    -------
    pandas.DataFrame
        A shuffled DataFrame containing synthetic instances generated from
        the input cluster datasets.
    """
    all_synthetic_data = []

    # Load all daily clusters
    daily_filepaths = get_all_filepaths(daily_dir)
    cluster_data_list = load_cluster_data(
        daily_filepaths + selected_weekly_files + selected_monthly_files
    )

    for df in cluster_data_list:
        # Generate as many synthetic samples as original
        num_samples = df.shape[0]
        synthetic_df = generate_synthetic_data(df, num_samples)
        all_synthetic_data.append(synthetic_df)

    synthetic_dataset = pd.concat(all_synthetic_data, ignore_index=True)
    synthetic_dataset = synthetic_dataset.sample(frac=1).reset_index(
        drop=True)  # Shuffle dataset
    return synthetic_dataset


# Function to save synthetic data
def save_synthetic_data(df, output_path):
    """
    Save the generated synthetic dataset to a specified file path in .pkl format.

    Parameters
    ----------
    df : pandas.DataFrame
        The synthetic dataset to be saved.
    output_path : str
        The file path where the dataset should be saved.
    """
    df.to_pickle(output_path)
    print(f"Synthetic dataset saved to {output_path}")


# Example usage
daily_dir = "../data/daily_clusters"
selected_weekly_files = [
    "../data/weekly_clusters/month_0_weekly_1.pkl",
    "../data/weekly_clusters/month_0_weekly_5.pkl",
    "../data/weekly_clusters/month_0_weekly_6.pkl",
    "../data/weekly_clusters/month_1_weekly_0.pkl",
    "../data/weekly_clusters/month_1_weekly_1.pkl",
    "../data/weekly_clusters/month_5_weekly_2.pkl",
    "../data/weekly_clusters/month_5_weekly_3.pkl"
]
selected_monthly_files = [
    "../data/monthly_clusters/cluster_4.pkl",
]

output_path = "../data/synthetic_smart_meters.pkl"

synthetic_data = create_synthetic_dataset(
    daily_dir,
    selected_weekly_files,
    selected_monthly_files
)

print(f"Shape of synthetic data: {synthetic_data.shape}")

save_synthetic_data(synthetic_data, output_path)
