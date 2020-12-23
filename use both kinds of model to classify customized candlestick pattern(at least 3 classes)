from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPool2D


def load_pkl(pkl_name):
    # load data from data folder
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
    return data

def get_cnn_model(params):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(10, 10, 4)))
    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    return model

def train_model(params, data):
    model = get_cnn_model(params)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    hist = model.fit(x=data['train_gaf'], y=data['train_label_arr'],
                     batch_size=params['batch_size'], epochs=params['epochs'], verbose=2)
    return (model, hist)

def print_result(data, model):
    # get train & test pred-labels
    train_pred = model.predict_classes(data['train_gaf'])
    test_pred = model.predict_classes(data['test_gaf'])
    # get train & test true-labels
    train_label = data['train_label'][:, 0]
    test_label = data['test_label'][:, 0]
    # confusion matrix
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(9))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(9))
    print(train_result_cm, '\n'*2, test_result_cm)
