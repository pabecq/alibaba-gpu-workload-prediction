import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#====== IMPORTING AND CLEANING THE DATA =====#
path = r'C:\Users\piere\My Drive\Cours\EDHEC\M2\thesis'


#----- USAGE DATA -----#
df_usage = (pd.read_csv(path + r'\usage.csv'))
print(df_usage.head())

df_usage['time'] = pd.to_datetime(df_usage['start_time'], unit='us')
df_usage.set_index('time', inplace=True)

df_usage_hourly = df_usage.resample("1h").mean().fillna(0)
df_usage_hourly.drop(['start_time'],axis=1, inplace=True)
df_usage_hourly.drop(['end_time'],axis=1, inplace=True)
print(df_usage_hourly)

#----- REQUEST DATA -----#
df_request = pd.read_csv(path + r'\request.csv')
print(df_request.head())

df_request['time']= pd.to_datetime(df_request['time'], unit='us')
df_request.set_index('time', inplace=True)

df_request_hourly = df_request.resample("1h").mean()
print(df_request_hourly)

#----- MERGING DATA -----#
df = pd.concat([df_usage_hourly, df_request_hourly], axis=1)
print(df.head())

df['cpu_waste'] = df['cpu_request'] - df['avg_cpu_usage']

fig, ax = plt.subplots()
plt.plot(df.index, df['avg_cpu_usage'])
plt.gcf().autofmt_xdate()
plt.show()


#====== FEATURE ENGINEERING =====#

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

dict_weekday = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
    }

df['day_of_week'].map(dict_weekday)


df['shift_1h'] = df['avg_cpu_usage'].shift(1)
df['shift_24h'] = df['avg_cpu_usage'].shift(24)

df['rolling_avg_12h'] = df['avg_cpu_usage'].rolling(12).mean()
df['rolling_avg_24h'] = df['avg_cpu_usage'].rolling(24).mean()
df['rolling_max_24h'] = df['avg_cpu_usage'].rolling(24).max()


#----- Removing NaN created by shifts and rolling features -----#
rows_before = len(df)
df = df.dropna()
rows_after = len(df)

#===== EXPORTING THE DATA =====#
output_file = path + r'\model_ready_data.csv'
df.to_csv(output_file)

