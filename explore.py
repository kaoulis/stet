from ydata_profiling import ProfileReport
import pandas as pd

asset = 'bal_sep_a'

df = pd.read_csv(f'data\\data_{asset}.csv', sep=';', decimal=",")
df['DateInterval'] = df['DateInterval'].apply(pd.to_datetime, errors='coerce')
df = df.fillna(method='ffill')

# Calculate the time difference in seconds between the current and the previous row
df['time_diff'] = (df['DateInterval'] - df['DateInterval'].shift()).dt.total_seconds()
# Create a new column for the operation number based on the time difference (probably we need more time.)
df['operationNo'] = (df['time_diff'] > 60).cumsum() + 1


##
df = df[['DateInterval',
         'a_BAL_SEPA_Feed_PV',
         'a_BAL_BOPA_Q_Carbon_Blower_Pressure',
         'a_BAL_BOPA_Q_Carbon_Pump_Current',
         'a_BAL_BOPA_Q_Mineral_Blower_Pressure',
         'a_BAL_BOPA_Q_Mineral_Pump_Current']]

df = df.dropna()
##


profile = ProfileReport(df, title="STET EDA", minimal=True)
profile.to_file(f"eda_{asset}.html")


## Explore for a specific operation
op_no = 79
df_op = df[df['operationNo'] == op_no]


##
profile = ProfileReport(df_op, title="STET EDA")
profile.to_file(f"op_{op_no}_eda_{asset}.html")
##

