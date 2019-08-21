import numpy as np
import pandas as pd
import pprint
import json
from datetime import datetime, time
from workalendar.asia import HongKong
from preprocessing import get_df
#----------------------------------------------------#
cleaned_df = get_df()
cal = HongKong()
start = time(9, 0)
end = time(18, 0)
def get_date_and_time(hk_time):
    date = datetime.strptime(hk_time.split()[0], "%Y-%m-%d") 
    time = datetime.strptime(hk_time.split()[1].split('+')[0], "%H:%M:%S").time()
    return (date, time)
def classify_time(hk_time):
    """ Given date, return whether it is a working day
    If working day, return if the time is during work or after work, assuming normal 9-6 hours 
    Otherwise, return False
    """
    # datetime object
    date, time = get_date_and_time(hk_time)
    is_working_day = cal.is_working_day(date)
    if is_working_day:
        return (is_working_day, start < time < end)
    else:
        return (is_working_day, False)
time_df = pd.DataFrame(columns=['user_id_str','working_day','working_hours'])
list_of_tuples = [classify_time(hk_time) for hk_time in cleaned_df['hk_time']]
time_df['user_id_str'] = cleaned_df['user_id_str']
time_df['working_day'], time_df['working_hours'] = list(zip(*list_of_tuples))
time_df.to_csv('time_analysis.csv')