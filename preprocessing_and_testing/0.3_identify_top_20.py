import numpy as np
import pandas as pd
df = pd.read_csv("initial_filtered.csv", low_memory=False, lineterminator="\n")
# get top 25 most active users
freq = df.user_id_str.value_counts()[1:101]
user_dict = {}
# create dictionary mapping id to tweets
for id in freq.index:
    user_dict[id] = df.loc[lambda row: row['user_id_str']==id]
new_df = pd.DataFrame(columns=df.columns)
for id in user_dict:
    new_df = new_df.append(user_dict[id].head(2))
new_df.to_csv("top_users.csv")