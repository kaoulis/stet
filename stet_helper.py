import numpy as np
import pandas as pd


def calculate_lags_windows(df_raw, op_features, target):
    df = df_raw.copy()
    df_corr = pd.DataFrame()
    df_mutual = pd.DataFrame()
    for lag in range(0, 120, 6):
        for window in range(0, 600, 60):
            df_shifted = df.copy()
            df_shifted[target] = df_shifted.groupby('continuous_period')[target].shift(-lag).dropna()

            df_shifted['sample_date'] = np.nan
            mask = ~df_shifted[target].isnull()
            df_shifted.loc[mask, 'sample_date'] = df_shifted.loc[mask, 'timestamp']
            df_shifted['sample_date'] = df_shifted.groupby('continuous_period')['sample_date'].fillna(method='bfill')
            df_shifted['sample_date'] = pd.to_datetime(df_shifted['sample_date'])
            df_shifted[target] = df_shifted.groupby('continuous_period')[target].fillna(method='bfill')
            df_shifted = df_shifted.dropna(subset=[target])

            df_shifted['sample_coming_in'] = (df_shifted['sample_date'] - df_shifted['timestamp']).dt.total_seconds()
            df_shifted['sample_period_no'] = (df_shifted['sample_coming_in'] > df_shifted['sample_coming_in'].shift()).cumsum() + 1

            df_shifted = df_shifted[df_shifted['sample_coming_in'] <= window]  # window style

            # Group the DataFrame by 'sample_period_no' and calculate mean and standard deviation for each feature
            grouped_df = df_shifted.groupby('sample_period_no')[op_features].agg(['mean'])
            grouped_df = grouped_df.stack().unstack(level=1)
            new_columns = [f'{col}_{stat}' for col, stat in grouped_df.columns]
            grouped_df.columns = new_columns
            grouped_df = grouped_df.reset_index()
            grouped_df = pd.merge(grouped_df, df_shifted[['sample_period_no', target]].drop_duplicates(), on='sample_period_no')
            grouped_df = grouped_df.dropna()

            corr_series = grouped_df.corr().loc[target, :].abs()
            corr_series.name = f'{lag}_{int(window / 10)}'
            df_corr = df_corr.append(corr_series)

            print(lag, window)

    return df_corr


def prepare_data(df_raw, op_features, target, criterion='r', const_lag=None, const_window=None):
    df = df_raw.copy()
    df['sample_date'] = np.nan
    mask = ~df[target].isnull()
    df.loc[mask, 'sample_date'] = df.loc[mask, 'timestamp']
    df['sample_date'] = df.groupby('continuous_period')['sample_date'].fillna(method='bfill')
    df['sample_date'] = pd.to_datetime(df['sample_date'])
    df[target] = df.groupby('continuous_period')[target].fillna(method='bfill')
    df = df.dropna(subset=[target])
    df['sample_coming_in'] = (df['sample_date'] - df['timestamp']).dt.total_seconds()
    df['sample_period_no'] = (df['sample_coming_in'] > df['sample_coming_in'].shift()).cumsum() + 1

    df_lag_window = pd.read_csv(f'corr_per_lag_window_{target}.csv', sep=';') if criterion == 'r' else pd.read_csv('mi_per_lag_window.csv')
    df_lag_window = df_lag_window[(~df_lag_window['Feature'].isin([target, 'sample_period_no'])) & (~df_lag_window['value'].isna())]

    for index, row in df_lag_window.iterrows():
        feature = row['Feature'].rsplit("_", 1)[0]
        aggr = row['Feature'].rsplit("_", 1)[1]
        lag = int(row['lag_window'].split("_")[0]) if const_lag is None else const_lag
        df[feature+'_'+aggr] = df.groupby('continuous_period')[feature].shift(lag)

    df = df.drop(op_features, axis=1)
    df = df.dropna()

    for index, row in df_lag_window.iterrows():
        feature = row['Feature'].rsplit("_", 1)[0]
        aggr = row['Feature'].rsplit("_", 1)[1]
        window = int(row['lag_window'].split("_")[1]) if const_window is None else const_window

        aggr_series = df[df['sample_coming_in'] <= window*10].groupby('sample_period_no').agg({feature+'_'+aggr: aggr}).squeeze()
        df[feature+'_'+aggr] = df['sample_period_no'].map(aggr_series)

    return df
