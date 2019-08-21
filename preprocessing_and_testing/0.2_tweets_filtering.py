import pandas as pd
import os
import numpy as np
import time
from datetime import datetime, timedelta
import pytz

import tweepy

# Check whether an account is bot
import botometer

from collections import Counter

mashape_key = "XXXXX"

twitter_app_auth = {
    'consumer_key': "XXXXX",
    'consumer_secret': "XXXXX",
    'access_token': "XXXXX",
    'access_token_secret': "XXXXX",
  }

bom = botometer.Botometer(wait_on_ratelimit=True,
                          mashape_key=mashape_key,
                          **twitter_app_auth)

# Hong Kong and Shanghai share the same time zone.
# Hence, we transform the utc time in our dataset into Shanghai time
time_zone_hk = pytz.timezone('Asia/Shanghai')


# Function used to output a pandas dataframe for each user based on the user account number
def derive_dataframe_for_each_user(df, all_users):
    dataframes = []
    for user in all_users:
        dataframes.append(df[df['user_id_str']==user])
    return dataframes


# Based on the dataframe for each user, compute the time range between his or her first tweet and last tweet
def compute_time_range_for_one_user(df):
    user_id_str = list(df['user_id_str'])[0]
    first_row = list(df.head(1)['created_at'])[0]
    end_row = list(df.tail(1)['created_at'])[0]
    datetime_object_first_row = datetime.strptime(first_row, '%a %b %d %H:%M:%S %z %Y')
    datetime_object_last_row = datetime.strptime(end_row, '%a %b %d %H:%M:%S %z %Y')
    time_range = datetime_object_last_row - datetime_object_first_row
    return (user_id_str, time_range.days)


# Check whether an account is bot or not based on the account number
# The id_str should be an integer
def check_bot(id_str):
    result = bom.check_account(int(id_str))
    return result['cap']['universal']


def delete_bots_have_same_geoinformation(df, saving_path, file_name, prop_threshold=0.70):
    users = set(list(df['user_id_str']))
    bot_account = []
    for user in users:
        dataframe = df.loc[df['user_id_str']==user]
        lat_counter = Counter(dataframe['lat'])
        lon_counter = Counter(dataframe['lon'])
        decide = (compute_the_highest_proportion_from_counter(lat_counter, prop_threshold)) or (compute_the_highest_proportion_from_counter(lon_counter, prop_threshold))
        # If only one unqiue geoinformation is found and more than 10 tweets are posted, we regard this account as bot
        if decide:
            bot_account.append(user)
        else:
            pass
    cleaned_df = df.loc[~df['user_id_str'].isin(bot_account)]
    cleaned_df.to_pickle(os.path.join(saving_path, file_name))
    return cleaned_df


def compute_the_highest_proportion_from_counter(counter_dict, prop_threshold):
    total_count = sum(counter_dict.values())
    result = False
    for latitude in list(counter_dict.keys()):
        if counter_dict[latitude]/total_count > prop_threshold:
            result = True
            return result
        else:
            pass
    return result


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


def get_month_hk_time(timestamp):
    """
    :param timestamp: timestamp variable after passing the pandas dataframe to add_eight_hours function
    :return: when the tweet is posted
    """
    month_int = timestamp.month
    if month_int == 1:
        result = 'Jan'
    elif month_int == 2:
        result = 'Feb'
    elif month_int == 3:
        result = 'Mar'
    elif month_int == 4:
        result = 'Apr'
    elif month_int == 5:
        result = 'May'
    elif month_int == 6:
        result = 'Jun'
    elif month_int == 7:
        result = 'Jul'
    elif month_int == 8:
        result = 'Aug'
    elif month_int == 9:
        result = 'Sep'
    elif month_int == 10:
        result = 'Oct'
    elif month_int == 11:
        result = 'Nov'
    else:
        result = 'Dec'
    return result


if __name__ == '__main__':

    start_time = time.time()
    print('Tweet filtering starts.....')
    whole_data = pd.read_csv("combined_csv.csv", low_memory=False, lineterminator="\n")
    # 1. Only consider the English and Chinese tweets
    whole_data_zh_en = whole_data.loc[whole_data['lang'].isin(['zh', 'en'])]
    # 2. Delete the verified accounts
    whole_data_without_verified = whole_data_zh_en.loc[whole_data_zh_en['verified'].isin([False])]
    # 3. Only keep the tweets which have geoinformation
    whole_data_geocoded = whole_data_without_verified.dropna(axis=0, subset=['lat'])
    whole_data_geocoded.to_csv("initial_filtered.csv")