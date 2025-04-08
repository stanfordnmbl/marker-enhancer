import numpy as np
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

# LSTM.
def get_lstm_model(input_dim, output_dim, nHiddenLayers, nHUnits, learning_r,
                   loss_f, bidirectional=False,
                   dropout=0, recurrent_dropout=0, weights=None):
    
    np.random.seed(1)
    tf.random.set_seed(1)

    model = Sequential()
    if bidirectional:
        model.add(Bidirectional(LSTM(units=nHUnits, return_sequences=True, 
                                     dropout=dropout, recurrent_dropout=recurrent_dropout), 
                                input_shape=(None, input_dim)))
    else:
        model.add(LSTM(units=nHUnits, input_shape = (None, input_dim),
                       return_sequences=True, dropout=dropout, 
                       recurrent_dropout=recurrent_dropout))

    if nHiddenLayers > 0:
        for i in range(nHiddenLayers):
            if bidirectional:
                model.add(Bidirectional(LSTM(units=nHUnits, 
                                             return_sequences=True,
                                             dropout=dropout,
                                             recurrent_dropout=recurrent_dropout)))
            else:
                model.add(LSTM(units=nHUnits, return_sequences=True,
                               dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(TimeDistributed(Dense(output_dim, activation='linear')))
    
    # Optimizer.
    opt=Adam(learning_rate=learning_r)
    
    # Loss function.
    if loss_f == "weighted_l2_loss":
        model.compile(
            optimizer=opt,
            loss=weighted_l2_loss(weights),
            metrics=[MeanSquaredError(), RootMeanSquaredError()])
    else:
        model.compile(
                optimizer=opt,
                loss=loss_f,
                metrics=[MeanSquaredError(), RootMeanSquaredError()])
    
    return model

# Linear.
def get_linear_regression_model(input_dim, output_dim, learning_r, loss_f, weights):
        
    np.random.seed(1)
    tf.random.set_seed(1)
    
    model = Sequential()
    model.add(Dense(output_dim, input_shape=(input_dim,)))
    
    # Optimizer.
    opt=Adam(learning_rate=learning_r)
    
    # Loss function.
    if loss_f == "weighted_l2_loss":
        model.compile(
            optimizer=opt,
            loss=weighted_l2_loss(weights),
            metrics=[MeanSquaredError(), RootMeanSquaredError()])    
    else:
        model.compile(
            optimizer=opt,
            loss=loss_f,
            metrics=[MeanSquaredError(), RootMeanSquaredError()])
    
    return model

# Weightedd L2 loss function.
def weighted_l2_loss(weights):
    def loss(y_true, y_pred):
        squared_diff = K.square(y_true - y_pred)
        weighted_squared_diff = squared_diff * weights
        return K.mean(weighted_squared_diff, axis=-1)
    return loss
