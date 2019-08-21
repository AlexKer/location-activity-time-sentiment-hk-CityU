import random
import os 
import csv
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocessing import read_local_csv, get_df
#----------------------------------------------------#
# Use LIWC and Vader, if both agree, classify, otherwise mark uncertain
def scale(posemo, negemo):
    """ Transform the output to a 0/1/2 result """
    if posemo > negemo:
        return 2
    elif negemo > posemo:
        return 0
    else:
        return 1
df_LIWC = read_local_csv('', 'LIWC2015_all_tweets_2018_cleaned.csv')
df_LIWC['y'] = df_LIWC.apply(lambda row: scale(row['posemo'], row['negemo']), axis = 1)

vader = SentimentIntensityAnalyzer()
def vader_polarity(text):
    """ Transform the output to a 0/1/2 result via Vadar recommended transformation"""
    score = vader.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 2
    elif (score['compound'] > -0.05) and (score['compound'] < 0.05):
        return 1
    else:
        return 0
df_vadar = get_df()
df_vadar['y'] = [vader_polarity(str(text)) for text in df_vadar['cleaned_text']]

compare = list(zip(df_LIWC['y'], df_vadar['y']))
combined = []
for a, b in compare:
    if a == b: 
        combined.append(a)
    else: 
        combined.append('uncertain')
# write aggregate predictions into new file
df_aggregate = pd.DataFrame()
df_aggregate['text'] = df_vadar['text']
df_aggregate['sentiment'] = combined
df_aggregate.to_csv("aggregate.csv")