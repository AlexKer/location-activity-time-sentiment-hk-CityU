import random
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score
analyser = SentimentIntensityAnalyzer()
#----------------------------------------------------#
# Source: https://nlpforhackers.io/sentiment-analysis-intro/
df = pd.read_csv("training_set.csv")

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

# training is not required for vadar, so not used
def prepare_data(text_col, label_col):
    sentiment_data = list(zip(df[text_col], df[label_col]))
    # randomly shuffle
    random.shuffle(sentiment_data)
    # apply 80/20 split
    train_X, train_Y = zip(*sentiment_data[:4000])
    test_X, test_Y = zip(*sentiment_data[4000:])
    return [train_X, train_Y, test_X, test_Y]

print('Vader')

sentiment_scheme1 = prepare_data("cleaned_text", "sentiment scheme1")
pred_y_train = [vader_polarity(text) for text in sentiment_scheme1[0]] # train_X
pred_y_test = [vader_polarity(text) for text in sentiment_scheme1[2]] # test_X
print("Scheme 1 accuracy %s" % accuracy_score(sentiment_scheme1[1]+sentiment_scheme1[3], pred_y_train+pred_y_test))
print("Scheme 1 f1 %s" % f1_score(sentiment_scheme1[1]+sentiment_scheme1[3], pred_y_train+pred_y_test, average='macro'))

sentiment_scheme2 = prepare_data("cleaned_text", "sentiment scheme2")
pred_y_train = [vader_polarity(text) for text in sentiment_scheme2[0]]
pred_y_test = [vader_polarity(text) for text in sentiment_scheme2[2]]
print("Scheme 2 accuracy %s" % accuracy_score(sentiment_scheme2[1]+sentiment_scheme2[3], pred_y_train+pred_y_test))
print("Scheme 2 f1 %s" % f1_score(sentiment_scheme2[1]+sentiment_scheme2[3], pred_y_train+pred_y_test, average='macro'))