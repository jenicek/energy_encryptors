import sys
from pathlib import Path
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
import matplotlib as mpl
mpl.use('Agg')
from plot import create_plots

pathReal = Path.cwd() / '../src/smart_meters_london_2013.csv'
pathSynth = Path.cwd() / f'../clusters/{sys.argv[1]}.csv'
print(pathSynth)
# pathSynth = pathReal

df_real = pd.read_csv(pathReal, parse_dates = ['timestamp']).set_index('timestamp')
df_synth = pd.read_csv(pathSynth, parse_dates = ['timestamp']).set_index('timestamp')

fig_dict, rmse_dict = create_plots(df_real, df_synth)

score = 0
for item in rmse_dict.values():
    score += item.loc[item['statistic'] != 'median', 'value'].sum()

print(score)


def get_features(df, label):
    df = df.astype("float32")
    df_features = extract_features(
        df.reset_index().melt(id_vars = 'timestamp', var_name = 'id', value_name = 'value'),
            column_id = 'id',
            column_sort = 'timestamp',
            column_value = 'value',
            n_jobs = 1,  # Use single process to avoid progress bar issues
            disable_progressbar = False,
            default_fc_parameters = {
                "absolute_sum_of_changes": None,
                "count_above_mean": None,
                "has_duplicate": None,
                "has_duplicate_max": None,
                "has_duplicate_min": None,
                "longest_strike_above_mean": None,
                "mean_abs_change": None,
                "percentage_of_reoccurring_values_to_all_values": None,
                "root_mean_square": None,
                "mean": None,
                "median": None,
                "skewness": None,
                "sum_values": None,
                "variance": None,
                "benford_correlation": None,
                "cid_ce": [{"normalize": True}],
                "fft_aggregated": [{"aggtype": "centroid"}],

            }
    )
    df_features['label'] = label
    return df_features


def create_df_features(df_real, df_synth):
    df_real_features = get_features(df_real, 1)
    df_synth_features = get_features(df_synth, 0)
    df_features = pd.concat([df_real_features, df_synth_features])
    df_features = df_features.loc[:, df_features.isna().sum() == 0]
    return df_features

df_features = create_df_features(df_real, df_synth)

X = df_features.drop('label', axis = 1).astype(float)
y = df_features['label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = xgb.XGBClassifier(eval_metric = 'logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy:.4f}')

featureImportance = model.feature_importances_
df_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': featureImportance
})
df_importance = df_importance.sort_values(by= 'Importance', ascending = False)
print(df_importance)
