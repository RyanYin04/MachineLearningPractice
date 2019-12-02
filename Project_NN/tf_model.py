import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder



class call_weight(keras.callbacks.Callback):
    '''
    This class is to help me track the wieght of the last layer using a 
    keras sequence
    '''
    # Def my own callback class so I can track the weight.
    def on_train_begin(self,logs = {}):
        self.weights = []
    def on_train_batch_end(self, batch, logs = {}):
        w = self.model.layers[2].get_weights()
        w_arr =[w[1][ 1]]
        w_arr.extend(list(w[0][:,1]))
        #self.weights.append(self.model.layers[2].get_weights())
        self.weights.append(w_arr)

class call_err(keras.callbacks.Callback):
    def on_train_begin(self, logs = {}):
        self.err_tr = []
        self.err_te = []
    
    def on_epoch_end(self,batch, logs = {}):
        pred_tr = self.model.predict(x_tr).argmax(axis = 1)
        pred_te = self.model.predict(x_te).argmax(axis = 1)
        err_tr, err_te = get_err(pred_tr, pred_te)
        self.err_tr.append(err_tr)
        self.err_te.append(err_te)
    
class call_loss(keras.callbacks.Callback):
    '''
    This class is to track losses.
    '''
    def on_train_begin(self, logs = {}):
        self.losses = []

    def on_train_batch_end(self, batch,  logs = {}):
        # print(logs)
        self.losses.append(logs.get('loss'))
                                         
# Build up the sequential model.  
def MLP(hidden_layers, nodes):
    MLP = Sequential()
    for l in range(hidden_layers):
        MLP.add(Dense(nodes, activation = tf.nn.sigmoid, 
        kernel_initializer = tf.initializers.RandomUniform ))
    MLP.add(Dense(10,activation = tf.nn.sigmoid, 
        kernel_initializer = tf.initializers.RandomUniform ))
    return MLP

MLP = MLP(2, 3)
# Initialize the trackers.
weight_tracker = call_weight()
loss_tracker = call_loss()
err_tracker = call_err()

# Compile and fit.
MLP.compile(optimizer = keras.optimizers.SGD(learning_rate = 3), loss = 'mse', metrics = ['accuracy'])

history = MLP.fit(x_tr,y_tr, epochs = 100, batch_size = 32,verbose = 2, callbacks = [weight_tracker, loss_tracker, err_tracker])


f = lambda x: 1/(1+math.exp(-x))
f(0)
f(0.6441136)
f(0.656)
f(0.658)
f(0.5)
f(0.62)
1.3*0.62*0.38*0.62