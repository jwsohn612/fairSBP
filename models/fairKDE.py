import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from tensorflow import keras

# The below code script is written by referring to the original work:   
# A Fair Classifier Using Kernel Density Estimation, 'Cho et al., 2020'.

class fairKDE(): 
    
    def __init__(self, loss_type, fair_type, out_act, learning_rate, dround, xdim, adim, plam, ldim):
        
        self.learning_rate = learning_rate 
        self.dround = dround 
        self.xdim = xdim
        self.adim = adim
        self.plam = plam
        self.ldim = ldim
        self.fair_type = fair_type 
        self.loss_type = loss_type
        self.out_act = out_act 
            
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.ce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
       
        delta = 1.0
        self.h = 0.02
        self.tau = 0.5
        self.huber = tf.keras.losses.Huber(delta=delta, name='huber_loss')
        
        self.define_optimizers()
        self.f = self.define_classifier()

    def fit_loss(self, y, yhat):
        if self.loss_type == 'mae':
            return self.mae(y, yhat)
        if self.loss_type == 'mse':
            return self.mse(y, yhat)
        if self.loss_type == 'ce':
            return self.ce(y, yhat)
            
    def define_optimizers(self):
        self.f_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)

    def define_classifier(self):
        h = keras.Sequential([
            layers.InputLayer(self.xdim,), 
            layers.Dense(self.ldim, activation=None), 
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(self.ldim, activation=None), 
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(self.ldim, activation=None), 
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(1, activation=self.out_act)
        ])
        return h

    def Q_function(self, x):
        a = 0.4920
        b = 0.2887
        c = 1.1893
        return tf.math.exp(-a*x**2 - b*x - c)
    
    def CDF_tau(self, Yhat):
        m = Yhat.shape[0]
        Y_tilde = (self.tau - Yhat)/self.h
        sum_ = tf.reduce_sum(self.Q_function(Y_tilde[Y_tilde>0])) + tf.reduce_sum(1 - self.Q_function(Y_tilde[Y_tilde<0])) + 0.5*(len(Y_tilde[Y_tilde==0]))
        return sum_/m
    
    def train_model(self, X, A, Y):
        if self.fair_type == 'sp':
            main_loss, huber_loss = self.train_model_dp(X, A, Y)
        elif self.fair_type == 'eo':
            main_loss, huber_loss = self.train_model_eo(X, A, Y)
        return main_loss, huber_loss
        
    def train_model_dp(self, X, A, Y):
        
        if self.adim == 1:
            with tf.GradientTape() as t:
                huber_loss = 0
                Yhat = self.f(X, training = True)
                PYhat =  self.CDF_tau(Yhat)
                main_loss = self.fit_loss(Y, Yhat)
                
                for j in [0,1]:
                    huber_loss += self.huber(tf.reshape((self.CDF_tau(Yhat[A[:,0]==j]) - PYhat),(-1)), 0) 

                total_loss = main_loss*(1-self.plam) + huber_loss * self.plam  
                grad = t.gradient(total_loss, self.f.trainable_variables)
                self.f_optim.apply_gradients(zip(grad, self.f.trainable_variables))

        elif self.adim == 2:
            with tf.GradientTape() as t:
                huber_loss = 0
                Yhat = self.f(X, training = True)
                PYhat =  self.CDF_tau(Yhat)
                main_loss = self.fit_loss(Y, Yhat)
                
                for l in [0,1]:
                    for j in [0,1]:
                        huber_loss += self.huber(tf.reshape((self.CDF_tau(Yhat[A[:,l]==j]) - PYhat),(-1)), 0) 
                        
                total_loss = main_loss*(1-self.plam) + huber_loss * self.plam 
                grad = t.gradient(total_loss, self.f.trainable_variables)
                self.f_optim.apply_gradients(zip(grad, self.f.trainable_variables))
            
        return main_loss, huber_loss
    
    
    def train_model_eo(self, X, A, Y):
        
        if self.adim == 1:
            with tf.GradientTape() as t:
                huber_loss = 0
                Yhat = self.f(X, training = True)
                main_loss = self.fit_loss(Y, Yhat)
                
                for y in [0,1]:
                    PYhat =  self.CDF_tau(Yhat[Y==y])
                    for r in [0,1]:
                        innerHat = Yhat[tf.reshape(Y==y,(-1))&(A[:,0]==r)]
                        if innerHat.shape[0] == 0:
                            continue
                        else:
                            PYhatZ = self.CDF_tau(innerHat)
                        huber_loss += self.huber(tf.reshape((PYhatZ - PYhat),(-1)), 0) 

                total_loss = main_loss*(1-self.plam) + huber_loss * self.plam  
                grad = t.gradient(total_loss, self.f.trainable_variables)
                self.f_optim.apply_gradients(zip(grad, self.f.trainable_variables))

        elif self.adim == 2:
            with tf.GradientTape() as t:
                huber_loss = 0
                Yhat = self.f(X, training = True)
                main_loss = self.fit_loss(Y, Yhat)
                
                for y in [0,1]:
                    PYhat =  self.CDF_tau(Yhat[Y==y])
                    for l in [0,1]:
                        for j in [0,1]:
                            innerHat = Yhat[tf.reshape(Y==y,(-1))&(A[:,0]==l)&(A[:,1]==j)]
                            if innerHat.shape[0] == 0:
                                continue
                            else:
                                PYhatZ = self.CDF_tau(innerHat)
                            huber_loss += self.huber(tf.reshape((PYhatZ - PYhat),(-1)), 0) 


                total_loss = main_loss*(1-self.plam) + huber_loss * self.plam  
                grad = t.gradient(total_loss, self.f.trainable_variables)
                self.f_optim.apply_gradients(zip(grad, self.f.trainable_variables))
            
        return main_loss, huber_loss   
    
    def evaluate(self, X):
        return self.f(X, training = False)