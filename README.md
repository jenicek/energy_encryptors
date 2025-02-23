# Energy Encryptors at the Watt's Up Hackathon

The Watt's Up Hackathon took place on February 22-23, 2025, in Vienna, where our team secured a third place.

![Team picture](energy_encryptors.jpg "Teamp picture")

## Reproduction of the submitted solution

1. Install dependencies in a virtual environment

```
python3 -m venv venv
pip install -r requirements.txt
```

2. Generate the synthetic data from real data

```
cd energy_encryptors/src
python3 generate_sequence.py
cd ../eval
python3 eval.py
```

⚠️ The scripts are not parametrized and rely on hard-coded paths

### Results of the submitted solutions

```
1.147897992895235
Test accuracy: 0.8448
                                              Feature  Importance
4                            value__has_duplicate_min    0.293615
5                    value__longest_strike_above_mean    0.158698
7   value__percentage_of_reoccurring_values_to_all...    0.077412
10                                      value__median    0.076222
0                      value__absolute_sum_of_changes    0.066261
1                             value__count_above_mean    0.064588
14                      value__cid_ce__normalize_True    0.059581
11                                    value__skewness    0.055487
13                                    value__variance    0.052959
8                             value__root_mean_square    0.049815
9                                         value__mean    0.045362
6                              value__mean_abs_change    0.000000
3                            value__has_duplicate_max    0.000000
2                                value__has_duplicate    0.000000
12                                  value__sum_values    0.000000
```

