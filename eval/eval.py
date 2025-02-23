from pathlib import Path
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')

from plot import create_plots

pathReal = Path.cwd() / '../src/smart_meters_london_2013.csv'
pathSynth = Path.cwd() / '../clusters/output2.csv'
# pathSynth = pathReal

df_real = pd.read_csv(pathReal, parse_dates = ['timestamp']).set_index('timestamp')
df_synth = pd.read_csv(pathSynth, parse_dates = ['timestamp']).set_index('timestamp')

fig_dict, rmse_dict = create_plots(df_real, df_synth)

score = 0
for item in rmse_dict.values():
    score += item.loc[item['statistic'] != 'median', 'value'].sum()

print(score)
