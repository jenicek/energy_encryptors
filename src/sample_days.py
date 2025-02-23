import pickle
import datetime
from multiprocessing import Pool
import pandas as pd
import numpy as np


def load_data():
    with open("../clusters/clusters.pkl", "rb") as handle:
        data = pickle.load(handle)
    return data

def sample_sequence(sequences):
    seq = np.zeros(sequences.shape[1], dtype=int)
    for i in range(sequences.shape[1]):
        if i == 0:
            choices = sequences[:,0]
        else:
            choices = sequences[sequences[:,i-1] == choice,i]
        choice = np.random.choice(choices)
        seq[i] = choice
    return seq

def random_sample(seq, cluster_members, days):
    series = np.zeros((seq.shape[0], 24))
    for i, c in enumerate(seq):
        choice = np.random.choice(cluster_members[c])
        series[i] = days[np.unravel_index(choice, days.shape[:2])]
    return series.reshape(-1)

def sample_time_series(seq, means, stds, covs):
    series = np.zeros((seq.shape[0], 24))
    for i, s in enumerate(seq):
        series[i] = [max(np.random.normal(x, y), 0.08) for x, y in zip(means[s], stds[s])]
        # series[i] = np.maximum(np.random.multivariate_normal(means[s], covs[s]), 0.08)
        # series[i] = means[seq[i]]
    return series.reshape(-1)

def synthesize(i):
    np.random.seed(i)
    seq = sample_sequence(data["sequences"])
    series = sample_time_series(seq, data["means"], data["stds"], data["covs"])
    # series = random_sample(seq, data["sequences"], data["days"])
    return series

data = load_data()
# output = np.zeros((4125, 24*365))
# for i in range(4125):
#     output[i] = synthesize(i)
with Pool(18) as p:
    output = p.map(synthesize, range(4125))
    output = np.vstack(output)

dataset = pd.read_csv('smart_meters_london_2013.csv')
dataset.iloc[:,1:] = output.T
dataset.to_csv('../clusters/output.csv', index=False)
