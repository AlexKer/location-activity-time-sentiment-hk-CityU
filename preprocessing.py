import pandas as pd
import os, csv
# two useful functions
def read_local_csv(path, filename, dtype_str=True):
    if dtype_str:
        dataframe = pd.read_csv(os.path.join(path, filename), encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index_col=0, dtype='str') 
    else:
        dataframe = pd.read_csv(os.path.join(path, filename), encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index_col=0) 
    return dataframe
# get cleaned df that we need
def get_df():
    return read_local_csv('', 'all_tweets_2018_cleaned.csv')