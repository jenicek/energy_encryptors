import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def load_data():
    dataset = pd.read_csv('smart_meters_london_2013.csv')
    df = dataset.T
    return df[1:].to_numpy().astype('float32')

data = load_data()
# data = data.reshape(data.shape[0], 365, 24)
days = data.reshape(-1, 24)

kmeans = KMeans(n_clusters=3000, random_state=0).fit(days)
clusters = kmeans.cluster_centers_
labels = kmeans.labels_
sequences = labels.reshape(data.shape[0], 365)

stds = np.zeros(clusters.shape)
covs = np.zeros(clusters.shape + (24,))
counts = np.zeros(clusters.shape[0], dtype=int)
for c, _ in enumerate(clusters):
    plt.figure()
    mask = labels.reshape(-1) == c
    errs = days[mask].std(axis=0)
    counts[c] = mask.sum()
    stds[c] = errs
    covs[c] = np.cov(days[mask].T)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.errorbar(list(range(24)), clusters[c], errs, linestyle='None', marker='.')
    plt.xlabel("hour of day")
    plt.ylabel("mean kWh consumntion")
    plt.title(f"Cluster {c} with {counts[c]} samples")
    plt.subplot(1, 2, 2)
    idx = np.random.choice(mask.nonzero()[0])
    plt.plot(days[idx])
    plt.title(f"Sample {np.unravel_index(idx, (data.shape[0], 365))}")
    plt.savefig(f'../clusters/cluster_{c}.png')
    plt.close()

cluster_samples = []
sequences1 = sequences.reshape(-1)
for s in range(len(clusters)):
    cluster_samples.append((sequences1 == s).nonzero()[0])

with open('../clusters/clusters.pkl', 'wb') as handle:
    pickle.dump({"means": clusters, "stds": stds, "covs": covs, "counts": counts,
                 "sequences": sequences, "days": days.reshape(data.shape[0], 365, 24),
                 "cluster_samples": cluster_samples}, handle)
