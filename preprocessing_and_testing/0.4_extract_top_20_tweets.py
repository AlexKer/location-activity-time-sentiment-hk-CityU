import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
#----------------------------------------------------#
df = pd.read_csv("initial_filtered.csv", low_memory=False, lineterminator="\n")
user_id_str_list = [30035884,53541705,160932283,14397810,883377606,14282045,148017981,52236952,36204274,9.37093E+17,7376602,191738379,62536337,53893147,7.25655E+17,96273306,37575049,1651882188,2399387766,55498924,16094512,129446245]
user_dict = {}
time_zone_hk = pytz.timezone('Asia/Shanghai')
#----------------------------------------------------#
# The time of tweet we have collected is recorded as the UTC time
# Add 8 hours to get the Hong Kong time
def get_hk_time(df):
    changed_time_list = []
    for _, row in df.iterrows():
        time_to_change = datetime.strptime(row['created_at'], '%a %b %d %H:%M:%S %z %Y')
        # get the hk time
        changed_time = time_to_change.astimezone(time_zone_hk)
        changed_time_list.append(changed_time)
    df['hk_time'] = changed_time_list
    return df
hk_df = get_hk_time(df)
#----------------------------------------------------#
# extract all tweets for each user and sort their tweets chronologically
# build sorted dictionary
for id in user_id_str_list:
    user_dict[id] = hk_df.loc[lambda row: row['user_id_str']==id].sort_values(by='hk_time') 
new_df = pd.DataFrame(columns=df.columns)
# construct df containing all their tweets
for id in user_dict:
    new_df = new_df.append(user_dict[id])
# delete tweets from China
new_df = new_df[new_df['country']!='People\'s Republic of China']
new_df.to_csv("all_top_users_tweets.csv")