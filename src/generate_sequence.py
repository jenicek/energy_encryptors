import pickle
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def load_data():
    dataset = pd.read_csv('smart_meters_london_2013.csv')
    df = dataset.T
    return dataset, df[1:].to_numpy().astype('float32')

dataset, data = load_data()
days = data.reshape(data.shape[0], 365, 24)

def generate_sequence(i):
    np.random.seed(i)
    sequence = np.zeros((365, 24))
    chosen = np.random.choice(days.shape[0])
    first_chosen = chosen
    sequence[0] = days[chosen,0]
    for i in range(1, 365):
        dist = distance.cdist(days[first_chosen,i].reshape(1, -1), days[:,i])
        sample = np.argpartition(dist.squeeze(), 4)[:5]
        chosen = sample[np.random.randint(1,5)]
        # sequence[i] = [round(max(np.random.normal(x, np.std(y)), 0.08 + np.random.normal(0, 0.01)), 3) for x, y in zip(days[chosen,i], days[sample[:2],i].T)]
        # sequence[i] = [np.random.lognormal(x, np.std(np.log(y))) for x, y in zip(days[chosen,i], days[sample,i].T)]
        sequence[i] = days[chosen,i]
    return sequence.reshape(-1)

# generate_sequence(0)
with Pool(18) as p:
    output = p.map(generate_sequence, range(4125))
    output = np.vstack(output)

dataset.iloc[:,1:] = output.T
dataset.to_csv('../clusters/output2.csv', index=False)
