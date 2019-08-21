import pandas as pd
import numpy as np
from preprocessing import read_local_csv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
#----------------------------------------------------#
# Classifcation with four labels: 0=negative, 1=neutral, 2=positive, 3=uncertain
# Using location type, activity, and time data as features
df = read_local_csv('dataset/', 'stats.csv')
# SMOTE (Synethic Minority Over-sampling Technique)
def perform_smote(X_train, y_train):
    smote = SMOTE('minority')
    X_sm, y_sm = smote.fit_sample(X_train, y_train)
    return X_sm, y_sm
# Try different dropout rates, epochs, batchsizes
dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
epochs_list = [10, 20, 30, 40, 50]
batch_size = [32, 64, 128]

# Shuffle data
df = df.reindex(np.random.permutation(df.index))
# Split into training and test sets
train, test = train_test_split(df, test_size=0.15)
# X features are set to all columns excluding the label
X_train = train[train.columns.difference(['sentiment'])]
y_train = train['sentiment']
X_test = test[train.columns.difference(['sentiment'])]
y_test = test['sentiment']

# Since feature vectors are sparse, we conduct PCA
# We have 111 features, will set n_components to 100, close to original num features
pca = PCA(n_components=100)
pca.fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()
# 40 components seems to be where the graph plateaus
NCOMPONENTS = 40
pca = PCA(n_components=NCOMPONENTS)
# Apply PCA to X_train and X_test
X_pca_train = pca.fit_transform(X_train)
X_pca_test = pca.fit_transform(X_test)

# MLP function
def run_MLP(X_train, y_train, X_test, y_test, input_dim, dropout, epochs, batch_size):
    print('-----------Running Multilayer Perceptron-----------')
    # Build model 
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))
    # Choose optimizer and loss function
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    loss = 'sparse_categorical_crossentropy'
    # Compile 
    model.compile(optimizer=opt, 
        loss=loss,
        metrics=['accuracy'])
    # Fit on training data and cross-validate
    model.fit(X_train, y_train,
        epochs=epochs,
        batch_size=batch_size)
    # Test on testing data
    score = model.evaluate(X_test, y_test, batch_size=batch_size)

# print(X_train.shape, X_sm.shape) #(3688 examples, 111 features)
# We need to reshape data from (3688, 111) to (3688, 111, 1)
def reshape_for_1DCNN(X):
    return np.expand_dims(X, axis=2)
X_train_CNN = reshape_for_1DCNN(X_train)
X_test_CNN = reshape_for_1DCNN(X_test)
X_pca_train_CNN = reshape_for_1DCNN(X_pca_train)
X_pca_test_CNN = reshape_for_1DCNN(X_pca_test)

# 1DCNN function
def run_1DCNN(X_train, y_train, X_test, y_test, input_dim, kernal_size, pool_size, dropout, epochs, batch_size):
    print('-----------Running 1D CNN-----------')
    # Build model 
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=kernal_size, activation='relu', input_shape=(input_dim, 1)))
    model.add(Conv1D(filters=128, kernel_size=kernal_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Choose optimizer and loss function
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    loss = 'sparse_categorical_crossentropy'
    # Compile 
    model.compile(optimizer=opt, 
        loss=loss,
        metrics=['accuracy'])
    # Fit on training data and cross-validate
    model.fit(X_train, y_train,
        epochs=epochs,
        batch_size=batch_size)
    # Test on testing data
    score = model.evaluate(X_test, y_test, batch_size=batch_size)

# TODO:
# List of hyperparameters to toggle: dropout, epochs, batchsizes
# For MLP: dense layers, neurons per layer
# For 1DCNN: filters, kernal size, strides, padding, pool size, num of convolution layers followed by pooling layers
run_MLP(X_pca_train, y_train, X_pca_test, y_test, 40, 0.1, 30, 128)
run_1DCNN(X_pca_train_CNN, y_train, X_pca_test_CNN, y_test, 40, 3, 3, 0.1, 30, 128)

# We artifically oversample after principal dimensionality reduction to see if we improve
# X_pca_train_sm, y_train_sm = perform_smote(X_pca_train, y_train)
# run_MLP(X_pca_train_sm, y_train_sm, X_pca_test, y_test, 40, 0.1, 30, 128)
