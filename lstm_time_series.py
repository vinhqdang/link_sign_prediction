import numpy as np
import pandas as pd
import urllib
from sklearn import preprocessing
import os

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

def run (index_file, epoch = 10):
    yt = load_data("node_distances/" + str(index_file))
    (x_train,y_train,x_test,y_test) = prepare_data(yt)
    pred, y_test = stateful_lstm(x_train, y_train, x_test, y_test, epoch=epoch)
    print (pred)
    print (y_test)

def load_data (file_name):
    """
    @brief      read a text file to return a vector of loat
    
    @param      file_name  The file name
    
    @return     { a vector of float }
    """
    if (os.path.isfile(file_name)):
        with open (file_name) as f:
            lines = f.readlines()
            res = []
            for line in lines:
                res += [float(x) for x in line.split()]
            return res
    return []

def prepare_data (yt):
    """
    @brief      {prepare data}
    
    @param      yt  an array
    
    @return     { description_of_the_return_value }
    """

    if len(yt) == 0:
        return 1
    if len(yt) <= 5:
        # return majority
        res = sum (x > 0 for x in yt)
        if res > 0:
            return 1
        else:
            return -1

    # add timelag data
    yt = pd.Series (yt)

    yt_1 = yt.shift (1)
    yt_2 = yt.shift (2)
    yt_3 = yt.shift (3)
    yt_4 = yt.shift (4)
    yt_5 = yt.shift (5)

    data = pd.concat ([yt, yt_1, yt_2, yt_3, yt_4, yt_5], axis = 1)

    data.columns = ['yt','yt_1','yt_2','yt_3','yt_4','yt_5']

    # drop NA cause by lag

    data = data.dropna ()
    y = data['yt']

    cols = ['yt_1','yt_2','yt_3','yt_4','yt_5']
    x = data[cols]

    print ('Preparing data finished.')

    print ('Preprocessing.')

    # scaler_x = preprocessing.MinMaxScaler (feature_range = (-1,1))

    x = np.array(x).reshape ((len(x),5))
    # x = scaler_x.fit_transform (x)

    # scaler_y = preprocessing.MinMaxScaler (feature_range = (-1,1))

    y = np.array(y).reshape ((len(y),1))
    # y = scaler_x.fit_transform (y)

    print ('Dividing train/test sets')

    assert (len(x) == len(y))

    train_end = len (y) - 1
    x_train = x[0:train_end,]
    x_test = x[train_end:len(x),]

    y_train = y[0:train_end]
    y_test = y [train_end:len(y)]

    x_train = x_train.reshape (x_train.shape + (1,))
    x_test = x_test.reshape (x_test.shape + (1,))

    return (x_train,y_train,x_test,y_test)

def stateful_lstm (x_train, y_train, x_test, y_test,
                    epoch=10):
    fit2 = Sequential ()
    fit2.add (LSTM (output_dim = 4,
                    stateful = True,
                    batch_input_shape=(1,5,1),
                    activation = 'tanh',
                    inner_activation = 'hard_sigmoid'))
    fit2.add (Dense (output_dim=1, activation='linear'))
    fit2.compile (loss = 'mean_squared_error', optimizer = 'rmsprop')

    end_point = len (x_train)
    start_point = 0

    for i in range (len (x_train[start_point:end_point])):
        print "Fitting example ",i
        fit2.fit (x_train[start_point:end_point], 
                    y_train[start_point:end_point],
                    nb_epoch = epoch,
                    batch_size = 1,
                    verbose = 2,
                    shuffle = True)
        fit2.reset_states ()

    print (fit2.summary())

    score_train = fit2.evaluate (x_train, y_train, batch_size=1)
    score_test = fit2.evaluate (x_test, y_test, batch_size=1)

    print ("train MSE = " + str(round(score_train,4)))
    print ("test MSE = " + str(round(score_test,4)))

    pred2 = fit2.predict (x_test, batch_size = 1)
    return (pred2, y_test)
    # pred2 = scaler_y.inverse_transform (np.array (pred2).reshape ((len(pred2),1)))

# fit1 = Sequential ()
# fit1.add (LSTM (output_dim=4,
#                 activation='tanh',
#                 inner_activation='hard_sigmoid',
#                 input_shape = (5,1)
#                 ))
# fit1.add (Dense (output_dim=1, activation='linear'))

# fit1.compile (loss = 'mean_squared_error', optimizer = 'rmsprop')

# fit1.fit (x_train, y_train, batch_size=1, nb_epoch=10, shuffle = True)

# print (fit1.summary())

# score_train = fit1.evaluate (x_train, y_train, batch_size=1)
# score_test = fit1.evaluate (x_test, y_test, batch_size=1)

# print ("train MSE = " + str(round(score_train,4)))
# print ("test MSE = " + str(round(score_test,4)))

# pred1 = fit1.predict (x_test)
# pred1 = scaler_y.inverse_transform (np.array(pred1).reshape((len(pred1),1)))

# stateful LSTM

