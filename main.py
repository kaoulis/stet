import dill as dill
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from helper import eda, evaluate, corr_plot, scatter_plot, rfr_feature_importance, ccf_values, FA
from stet_helper import calculate_lags_windows, prepare_data

class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name, lambda_=1):
        self.feature_name = feature_name
        self.lambda_ = lambda_

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X[self.feature_name] = stats.boxcox(X[self.feature_name].to_numpy(), lmbda=self.lambda_)
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X[self.features]
        return X

## Load raw data, eda, and drop useless columns
quality_features = ['a_BAL_SEPA_Q_C_FI', 'a_BAL_SEPA_Q_C_LOI', 'a_BAL_SEPA_Q_F_FI', 'a_BAL_SEPA_Q_F_LOI', 'a_BAL_SEPA_Q_P_FI', 'a_BAL_SEPA_Q_P_LOI',
                    'a_BAL_SEPA_Q_PC_LOI', 'a_BAL_SEPA_Q_Y']
operational_features = ['a_BAL_SEPA_ACarBlwr\PressX10', 'a_BAL_SEPA_ACarPump\Ampsx10', 'a_BAL_SEPA_AMinBlwr\PressX10', 'a_BAL_SEPA_AMinPump\Ampsx10',
                        'a_BAL_SEPA_A_HV_Neg_kV_PV', 'a_BAL_SEPA_A_HV_Neg_mA_PV', 'a_BAL_SEPA_A_HV_Pos_kV_PV', 'a_BAL_SEPA_A_HV_Pos_mA_PV',
                        'a_BAL_SEPA_AmbientTemp', 'a_BAL_SEPA_A_Belt_Gap', 'a_BAL_SEPA_a_Speed_PV', 'a_BAL_SEPA_E1_Torque_PV',
                        'a_BAL_SEPA_E2_Torque_PV', 'a_BAL_SEPA_AAshFeeder\PVx10', 'a_BAL_SEPA_Feed_RH', 'a_BAL_SEPA_feed_Temp',
                        'a_BAL_SEPA_Slave_Belt_Speedx10']

stet_raw_path = 'C:/Users/k.chiotis/OneDrive - Titan Cement Company SA/Desktop/STET/BAL SEP A/2023_05_11/data_bal_sep_a.csv'
stet_tags_map_path = 'C:/Users/k.chiotis/OneDrive - Titan Cement Company SA/Desktop/STET/BAL SEP A/Historian Kepware Tag mapping.csv'
stet_tags_path = 'C:/Users/k.chiotis/PycharmProjects/Assets/bal_stet/backup/tags.csv'

df_tags = pd.read_csv(stet_tags_path, sep=';', decimal=",")
df_tags = df_tags[df_tags['property'] == 'pv']
tags = pd.Series(df_tags.variable_name.values, index=df_tags.name).to_dict()

df_tags_map = pd.read_csv(stet_tags_map_path, sep=';')
tags_map = pd.Series(df_tags_map['OPC Tag Name'].values, index=df_tags_map['Historian Tag Name']).to_dict()

df_raw = pd.read_csv(stet_raw_path, sep=';', decimal=",")

df_raw = df_raw.rename(columns={'DateInterval': 'timestamp'})
df_raw = df_raw.rename(columns=tags_map)

df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

df_raw['a_BAL_SEPA_AAshFeeder\PVx10'] = df_raw['a_BAL_SEPA_AAshFeeder\PVx10'].fillna(df_raw['a_BAL_SEPA_Feed_PV'])

df_raw = df_raw[['timestamp'] + [x for x in df_tags['name'].values if x in df_raw.columns]]

df_raw = df_raw.loc[:, df_raw.columns.notna()]

df_raw = df_raw.drop(columns=['a_BAL_SEPA_AmbientRH', 'a_BAL_SEPA_ATransAir_RH', 'a_BAL_SEPA_ATransAir_Temp', 'a_BAL_SEPA_M1_Speed_PV'])

## Clean data: impute 10sec rows of 1min diff ; ffill non quality features; filter nwc data; feature selection
df_clean = df_raw.copy()
df_clean.set_index('timestamp', drop=False, inplace=True)

df_filtered = df_clean[((df_clean['timestamp'].shift(-1) - df_clean['timestamp']).dt.total_seconds() == 60) &
                       ((df_clean['timestamp'].shift(-2) - df_clean['timestamp']).dt.total_seconds() == 120) &
                       ((df_clean['timestamp'].shift(-3) - df_clean['timestamp']).dt.total_seconds() == 180)]

df_filtered = df_filtered.drop(columns=['a_BAL_SEPA_Q_Y', 'a_BAL_SEPA_Q_P_LOI'])
df_filtered = df_filtered.resample('10S').bfill()
df_filtered = pd.merge(df_filtered, df_clean[['a_BAL_SEPA_Q_Y', 'a_BAL_SEPA_Q_P_LOI']], left_index=True, right_index=True, how='left')

df_clean.drop(columns=['timestamp'], inplace=True)
df_filtered.drop(columns=['timestamp'], inplace=True)

df_clean = pd.concat([df_clean, df_filtered]).sort_index()
df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
df_clean['timestamp'] = df_clean.index
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])

df_clean = df_clean.replace({'a_BAL_SEPA_Q_Y': {0: np.nan},
                             'a_BAL_SEPA_Q_P_LOI': {0: np.nan}})
df_clean.loc[:, ~df_clean.columns.isin(quality_features)] = df_clean.loc[:, ~df_clean.columns.isin(quality_features)].fillna(method='ffill')

df_clean = df_clean[df_clean['a_BAL_SEPA_AAshFeeder\PVx10'] > 150]
df_clean = df_clean[df_clean['a_BAL_SEPA_A_Belt_Gap'] < 550]
df_clean = df_clean[df_clean['a_BAL_SEPA_A_HV_Neg_mA_PV'] > 100]
df_clean = df_clean[df_clean['a_BAL_SEPA_A_HV_Pos_mA_PV'] > 100]

df_clean['continuous_period'] = ((df_clean['timestamp'] - df_clean['timestamp'].shift()).dt.total_seconds() > 10).cumsum() + 1
group_sizes = df_clean.groupby('continuous_period').transform('size')
df_clean = df_clean[group_sizes >= 100]

# df_clean['gap_type'] =

## Inspect lags and windows per feature
target = 'a_BAL_SEPA_Q_P_LOI' # a_BAL_SEPA_Q_P_LOI, a_BAL_SEPA_Q_Y
# df_corr = calculate_lags_windows(df_clean, operational_features, target)
#
# pd.DataFrame({'Feature': df_corr.max().index,
#               'lag_window': df_corr.idxmax().values,
#               'value': df_corr.max().values}
#              ).sort_values(by='value', ascending=False).to_csv(f'corr_per_lag_window_{target}.csv')

##
## Train Yield
##
df_ready = prepare_data(df_clean.drop(columns=['a_BAL_SEPA_Q_P_LOI']), operational_features, 'a_BAL_SEPA_Q_Y', criterion='r')
df_ready = df_ready.drop_duplicates(subset=['sample_period_no'])
df_ready = df_ready.dropna()
df_ready.columns = df_ready.columns.str.replace('_mean$', '')

##
df1 = df_ready.drop(['timestamp', 'sample_date', 'sample_coming_in', 'sample_period_no', 'continuous_period'], axis=1)
# df1 = df1.drop([f+'_mean' for f in operational_features] + [f+'_std' for f in operational_features], axis=1, errors='ignore')
y = df1['a_BAL_SEPA_Q_Y'].copy()
X = df1.drop(['a_BAL_SEPA_Q_Y'], axis=1)

X_train = X[:int(0.75 * len(X))]
X_test = X[int(0.75 * len(X)):]
y_train = y[:int(0.75 * len(y))]
y_test = y[int(0.75 * len(y)):]

pipe = Pipeline([
    ('inputs', FeatureSelector(X_train.columns)),
    # ('scale', StandardScaler()),
    # ('pca', PCA()),
    # ('fa', FA()),
    ('pls', PLSRegression()),
    # ('model', None)
])
param_grid = {
    'pls__n_components': [i for i in range(4, X_train.shape[1])],
    'pls__scale': [True, False],
    'pls__max_iter': [i for i in range(100, 1000, 100)],
    # 'pca__n_components': [i for i in range(4, X_train.shape[1])],
    # 'fa__n_factors': [i for i in range(4, X_train.shape[1])],
    # 'fa__rotation': [None, 'varimax', 'promax'],
    # 'model': [RandomForestRegressor()],
}

tscv = TimeSeriesSplit(n_splits=2, test_size=30)
for train_index, test_index in tscv.split(X_train):
    print('Train set: ', train_index[0], ' - ', train_index[-1])
    print('Test set: ', test_index[0], ' - ', test_index[-1])
    print('~~~~~~~~~~~~~~~~~~')


##
grid = GridSearchCV(pipe,
                    param_grid=param_grid,
                    cv=tscv,
                    n_jobs=-1,
                    refit='r2',
                    scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'])
# grid.fit(X, y)
grid.fit(X_train, y_train)
df_grid = pd.DataFrame(grid.cv_results_).sort_values('rank_test_r2')

## Evaluate model
evaluate(grid, X_test, y_test, df1)
coef_df = pd.DataFrame({'feature': X_train.columns, 'coefficient': grid.best_estimator_.named_steps['pls'].coef_.flatten().tolist()})
print(coef_df.sort_values(by='coefficient', ascending=False))

## Train on the entire data
grid.fit(X, y)
df_grid = pd.DataFrame(grid.cv_results_).sort_values('rank_test_r2')
with open('Product Yield.sav', 'wb') as f:
    dill.dump(grid.best_estimator_, f)

##
df_lag_window = pd.read_csv(f'corr_per_lag_window_a_BAL_SEPA_Q_Y.csv', sep=';')
df_lag_window['Feature'] = df_lag_window['Feature'].str.replace('_mean$', '')

df_model_input = pd.DataFrame({'feature': X_train.columns, 'coefficient': grid.best_estimator_.named_steps['pls'].coef_.flatten().tolist()})
df_model_input = df_model_input.merge(df_tags[['variable_name', 'name']], how='left', left_on='feature', right_on='name')
df_model_input = df_model_input.merge(df_lag_window[['Feature', 'lag_window']], how='left', left_on='feature', right_on='Feature')
df_model_input['lag'] = df_model_input['lag_window'].str.split('_').str[0]
df_model_input['window'] = df_model_input['lag_window'].str.split('_').str[1]
df_model_input.to_csv('asdf.csv', sep=';')

##
## Train LOI
##
df_ready = prepare_data(df_clean.drop(columns=['a_BAL_SEPA_Q_Y']), operational_features, 'a_BAL_SEPA_Q_P_LOI', criterion='r')
df_ready = df_ready.drop_duplicates(subset=['sample_period_no'])
df_ready = df_ready.dropna()
df_ready.columns = df_ready.columns.str.replace('_mean$', '')
_, lambda_Neg = stats.boxcox(df_ready['a_BAL_SEPA_A_HV_Neg_mA_PV'].to_numpy())
_, lambda_Pos = stats.boxcox(df_ready['a_BAL_SEPA_A_HV_Pos_mA_PV'].to_numpy())
# df_ready = df_ready[np.abs(df_ready['a_BAL_SEPA_Q_P_LOI'] - df_ready['a_BAL_SEPA_Q_P_LOI'].mean()) <= (3 * df_ready['a_BAL_SEPA_Q_P_LOI'].std())]

##
df1 = df_ready.drop(['timestamp', 'sample_date', 'sample_coming_in', 'sample_period_no', 'continuous_period'], axis=1)
# df1 = df1.drop([f+'_mean' for f in operational_features] + [f+'_std' for f in operational_features], axis=1, errors='ignore')
y = df1['a_BAL_SEPA_Q_P_LOI'].copy()
X = df1.drop(['a_BAL_SEPA_Q_P_LOI'], axis=1)

X_train = X[:int(0.75 * len(X))]
X_test = X[int(0.75 * len(X)):]
y_train = y[:int(0.75 * len(y))]
y_test = y[int(0.75 * len(y)):]

pipe = Pipeline([
    ('inputs', FeatureSelector(X_train.columns)),
    ('boxcox1', BoxCoxTransformer('a_BAL_SEPA_A_HV_Neg_mA_PV', lambda_Neg)),
    ('boxcox2', BoxCoxTransformer('a_BAL_SEPA_A_HV_Pos_mA_PV', lambda_Pos)),
    # ('scale', StandardScaler()),
    # ('pca', PCA()),
    # ('fa', FA()),
    ('pls', PLSRegression()),
    # ('model', None)
])
param_grid = {
    'pls__n_components': [i for i in range(4, X_train.shape[1])],
    'pls__scale': [True, False],
    'pls__max_iter': [i for i in range(100, 1000, 100)],
    # 'pca__n_components': [i for i in range(4, X_train.shape[1])],
    # 'fa__n_factors': [i for i in range(4, X_train.shape[1])],
    # 'fa__rotation': [None, 'varimax', 'promax'],
    # 'model': [RandomForestRegressor()],
}

tscv = TimeSeriesSplit(n_splits=2, test_size=30)
for train_index, test_index in tscv.split(X_train):
    print('Train set: ', train_index[0], ' - ', train_index[-1])
    print('Test set: ', test_index[0], ' - ', test_index[-1])
    print('~~~~~~~~~~~~~~~~~~')


##
grid = GridSearchCV(pipe,
                    param_grid=param_grid,
                    cv=tscv,
                    n_jobs=-1,
                    refit='r2',
                    scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'])
# grid.fit(X, y)
grid.fit(X_train, y_train)
df_grid = pd.DataFrame(grid.cv_results_).sort_values('rank_test_r2')

##
evaluate(grid, X_test, y_test, df1)
coef_df = pd.DataFrame({'feature': X_train.columns, 'coefficient': grid.best_estimator_.named_steps['pls'].coef_.flatten().tolist()})
print(coef_df.sort_values(by='coefficient', ascending=False))

##
grid.fit(X, y)
df_grid = pd.DataFrame(grid.cv_results_).sort_values('rank_test_r2')
with open('Product LOI.sav', 'wb') as f:
    dill.dump(grid.best_estimator_, f)

##
df_lag_window = pd.read_csv(f'corr_per_lag_window_a_BAL_SEPA_Q_P_LOI.csv', sep=';')
df_lag_window['Feature'] = df_lag_window['Feature'].str.replace('_mean$', '')

df_model_input = pd.DataFrame({'feature': X_train.columns, 'coefficient': grid.best_estimator_.named_steps['pls'].coef_.flatten().tolist()})
df_model_input = df_model_input.merge(df_tags[['variable_name', 'name']], how='left', left_on='feature', right_on='name')
df_model_input = df_model_input.merge(df_lag_window[['Feature', 'lag_window']], how='left', left_on='feature', right_on='Feature')
df_model_input['lag'] = df_model_input['lag_window'].str.split('_').str[0]
df_model_input['window'] = df_model_input['lag_window'].str.split('_').str[1]
df_model_input.to_csv('asdf.csv', sep=';')

## Save model as a sav file
# with open('Product Yield.sav', 'wb') as f:
#     dill.dump(grid.best_estimator_, f)
