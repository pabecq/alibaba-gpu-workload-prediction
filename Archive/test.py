import pandas as pd
import ast

# ==== # IMPORT DATA # ==== #
input_path = r'C:\Users\piere\My Drive\Cours\EDHEC\M2\thesis\borg_traces_data.csv'
output_path = r'C:\Users\piere\My Drive\Cours\EDHEC\M2\thesis\model_ready_data_2.csv'

print("Loading Kaggle dataset...")
df = pd.read_csv(input_path)
print(f"Raw data loaded: {len(df)} rows")


# ==== # CONVERT TIME IN UNDERSTANDABLE FORMAT # ==== #
df['time'] = pd.to_datetime(df['start_time'], unit='us')


# ==== # EXTRACT CPU AND MEMORY USAGE # ==== #
print("Extracting CPU and Memory usage")
DICT_COLUMNS = ['resource_request', 'average_usage', 'maximum_usage']
def safe_literal_eval(value):
    """Converts string representation of dict/list to Python object."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If parsing fails, return the original string or None/{} depending on need
            return {} # Returning empty dict ensures the .apply(pd.Series) step doesn't fail
    return value
df[DICT_COLUMNS] = df[DICT_COLUMNS].map(safe_literal_eval)

for col_name in DICT_COLUMNS:
    temp_df = df[col_name].apply(pd.Series)
    temp_df.columns = [f"{col_name}_{key}" for key in temp_df.columns]
    df = pd.concat([df, temp_df], axis=1)
    df = df.drop(columns=[col_name])

df.head()

# --- INSERT THIS BLOCK AFTER YOUR DICT_COLUMNS LOOP ---

# Explicitly ensure the critical columns are treated as floating point numbers
# This is required because they were created from strings (objects)
COLUMNS_TO_CONVERT = [
    'average_usage_cpus',
    'maximum_usage_cpus',
    'resource_request_cpus'
]

# Use .astype(float) to force the conversion, using errors='coerce' to turn any remaining junk
# (like empty strings) into NaN, which the imputer will later handle.
for col in COLUMNS_TO_CONVERT:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows that failed conversion (e.g., if the raw entry was corrupted text)
df.dropna(subset=COLUMNS_TO_CONVERT, inplace=True)

print("Successfully converted core usage/request columns to float type.")

# ==== # FIND SPECIMENS THAT ARE ACTIVE AND HAVE A LONG ENOUGH USAGE # ==== #
print("Finding the best specimens for the analysis")
stats = df.groupby(['collection_id', 'instance_index'])['maximum_usage_cpus'].agg(['count', 'max'])
valid_candidates = stats[
    (stats['count'] > 100) &
    (stats['max'] > 0.005)
    ].sort_values('count', ascending=False)

print(valid_candidates.count())


top_10_instances = valid_candidates.head(10).index
print(f"Selected Specimen: Collection {top_10_instances}, Instance {top_10_instances}")
print(f"Stats: {valid_candidates.head(10).index}")


# --- 3. EXECUTION: LOOP, PROCESS, AND COMBINE (Panel Data Creation) ---
global_data_list = []

# The top_10_instances list contains Tuples of (collection_id, instance_index)
for collection_id, instance_index in top_10_instances:
    # 3a. Filter the raw data for the current specimen
    df_specimen = df[
        (df['collection_id'] == collection_id) &
        (df['instance_index'] == instance_index)
        ].copy()

    # Create the unique categorical ID for this specimen
    instance_id = f"{collection_id}_{instance_index}"

    # --- Start Processing for the Current Instance ---

    # Set time index and sort
    df_specimen.set_index('time', inplace=True)
    df_specimen.sort_index(inplace=True)

    # Resample to hourly: Usage=Mean(fillna 0), Request=Max(ffill)
    df_hourly = pd.DataFrame()

    # NOTE: The columns are named 'average_usage_cpus' and 'resource_request_cpus'
    # based on the new parsing logic. We must use these names!
    df_hourly['max_cpu_usage'] = df_specimen['maximum_usage_cpus'].resample('1H').max().fillna(0)
    df_hourly['cpu_request'] = df_specimen['resource_request_cpus'].resample('1H').max().ffill()

    # Final cleanup of request column (for clean start)
    df_hourly['cpu_request'] = df_hourly['cpu_request'].bfill().ffill()

    # --- 4. Feature Engineering ---
    target = 'max_cpu_usage'

    # Calendar Features
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['day_of_week'] = df_hourly.index.dayofweek
    df_hourly['is_weekend'] = df_hourly['day_of_week'].isin([5, 6]).astype(int)

    # Lag and Rolling Features (applied to this specific time series)
    df_hourly['shift_1h'] = df_hourly[target].shift(1)
    df_hourly['shift_24h'] = df_hourly[target].shift(24)
    df_hourly['rolling_avg_12h'] = df_hourly[target].shift(1).rolling(window=12).mean()
    df_hourly['rolling_avg_24h'] = df_hourly[target].shift(1).rolling(window=24).mean()
    df_hourly['rolling_max_24h'] = df_hourly[target].shift(1).rolling(window=24).max()

    # --- Add Categorical ID and Finalize ---
    df_hourly['instance_id'] = instance_id

    # Drop NaNs created by lag features
    df_hourly.dropna(inplace=True)

    # Append to the global list
    global_data_list.append(df_hourly)

# --- 5. COMBINE AND SAVE ---
if not global_data_list:
    print("Error: No valid time series were processed.")
else:
    # Concatenate all 10 instances into one DataFrame
    df_global = pd.concat(global_data_list)

    # Final check and save
    df_global['cpu_waste'] = df_global['cpu_request'] - df_global['max_cpu_usage']
    df_global.to_csv(output_path)

    print("\n--- GLOBAL MODEL DATA CREATED ---")
    print(f"Total Rows: {len(df_global)}")
    print(f"Total Instances: {df_global['instance_id'].nunique()}")
    print(f"Saved to: {output_path}")