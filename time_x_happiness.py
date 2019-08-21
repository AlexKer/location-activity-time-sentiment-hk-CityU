import numpy as np
import pandas as pd
from preprocessing import read_local_csv
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import calendar
#----------------------------------------------------#
df = read_local_csv('', 'final_dataset.csv')
def get_weekday_and_hour(hk_time):
    weekday = datetime.strptime(hk_time.split()[0], "%Y-%m-%d") .weekday()
    hour = datetime.strptime(hk_time.split()[1].split('+')[0], "%H:%M:%S").time().hour
    return (weekday, hour)
df['weekday'] = ''
df['hour'] = ''
# Visualize relationship between hours in a week, and its associated sentiment
for index, row in df.iterrows():
    # Monday is 0 and Sunday is 6
    row['weekday'], row['hour'] = get_weekday_and_hour(row['hk_time'])
weekday_dict = {0:None,1:None,2:None,3:None,4:None,5:None,6:None}
# Group all weekdays together and sort by hour
for weekday in weekday_dict.keys():
    weekday_dict[weekday] = df.loc[lambda row: row['weekday']==weekday].sort_values(by='hour') 
new_df = pd.DataFrame(columns=df.columns)
for weekday in weekday_dict.keys():
    new_df = new_df.append(weekday_dict[weekday])
new_df = new_df[['sentiment','weekday','hour']]
# First get rid of uncertain observations
new_df = new_df[new_df.sentiment!='uncertain']
new_df['weekday_hour'] = ''
for index, row in new_df.iterrows():
    row['weekday_hour'] = str(row['weekday'])+'_'+str(row['hour'])
new_df.to_csv('time_x_happiness.csv')
# Now attempt to plot time series data
# Get a mean sentiment value for each hour, of each weekday
# The list of hourly sentiment should be 24 hours/day * 7 days/week = 168 entries long
weekday_hour_list = [] # x
hourly_sentiment_list = [] # y
# Compute hourly sentiment via polarity measure
def compute_pos_minus_neg_percent(list_of_values):
    counter_dict = Counter(list_of_values)
    if len(list_of_values) != 0:
        result = (counter_dict[2] - counter_dict[0])/len(list_of_values)
    else:
        result = 0
    return result
# Perform calculations
for weekday in range(0, 7):
    for hour in range(0, 24):
        current = str(weekday)+'_'+str(hour)
        weekday_hour_list.append(current)
        # This is for one particular weekday and one particular hour
        cur_df = new_df.loc[new_df['weekday_hour']==current]
        cur_df['sentiment'] = pd.to_numeric(cur_df['sentiment'])
        hourly_sentiment_list.append(compute_pos_minus_neg_percent(cur_df['sentiment']))
# *Labels and tick marks of matplotlib needs to be seperated
# Plot graph
x = list(range(len(hourly_sentiment_list)))
fig, ax = plt.subplots(1,1, figsize=(20, 8))
ax.plot(x, hourly_sentiment_list)
plt.show()
# Examine what the specific parts of fluctuations
result_df = pd.DataFrame(columns=['Time', 'Sentiment'])
result_df['Time'] = weekday_hour_list
result_df['Sentiment'] = hourly_sentiment_list
# Output "weekday, hour" given a "weekdayNum_hourNum" input
d = dict(zip(range(7), calendar.day_name))
def get_English_time(number):
    return d[int(number.split('_')[0])]+', '+number.split('_')[1]+":00"
# Where the really unhappy hour is
print([get_English_time(i) for i in result_df.loc[result_df['Sentiment'] < -0.2].Time])
# And the really happy hour
print([get_English_time(i) for i in result_df.loc[result_df['Sentiment'] > 0.8].Time])