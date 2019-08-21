import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
# combined files containing translation and correct labels
df = pd.read_csv("LIWC2015 Results (training_set).csv")
def scale(posemo, negemo):
    """ Transform the output to a 0/1/2 result """
    if posemo > negemo:
        return 2
    elif negemo > posemo:
        return 0
    else:
        return 1
df['affect'] = df.apply(lambda row: scale(row['posemo'], row['negemo']), axis = 1)
# accuracy(groudtruth, label)
scheme1 = df['sentiment scheme1']
scheme2 = df['sentiment scheme2']
label = df['affect']
print("LIWC 2015")
print("Scheme 1 accuracy %s" % accuracy_score(scheme1, label))
print("Scheme 1 f1 %s" % f1_score(scheme1, label, average='macro'))
print("Scheme 2 accuracy %s" % accuracy_score(scheme2, label))
print("Scheme 2 f1 %s" % f1_score(scheme2e, label, average='macro'))