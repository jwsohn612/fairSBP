import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from tensorflow import keras



class fairNFP(): 
    
    def __init__(self, loss_type, fair_type, out_act, learning_rate, dround, xdim, adim, plam, ldim):
        
        self.learning_rate = learning_rate 
        self.dround = dround 
        self.xdim = xdim
        self.adim = adim
        self.ydim = 1
        self.plam = plam
        self.ldim = ldim
        self.sdim = int(self.ldim)
        self.fair_type = fair_type 
        self.loss_type = loss_type
        
        self.out_act = out_act 
        self.sout_act = 'sigmoid'
        self.clip_norm = False
        act_out = 'sigmoid'
            
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.ce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        
        self.define_optimizers()
        self.D = self.define_discriminator()
        self.h = self.define_classifier()
        self.pD = self.define_prediscriminator()

    def da(self, x):
        return tf.math.sigmoid(x)

    def neural_loss(self, x1,x2):
        real_loss = self.ce(tf.ones_like(x1), x1)
        fake_loss = self.ce(tf.zeros_like(x2), x2)
        D_loss = real_loss + fake_loss
        return  D_loss
    
    def neural_wloss(self, x1, x2, weight):
        loss = tf.math.log(x1+1e-5) + tf.multiply(tf.math.log(1-x2+1e-5),weight)
        return -tf.reduce_mean(loss)
    
    def fit_loss(self, y, yhat):
        if self.loss_type == 'mae':
            return self.mae(y, yhat)
        if self.loss_type == 'mse':
            return self.mse(y, yhat)
        if self.loss_type == 'ce':
            return self.ce(y, yhat)
            
    def define_optimizers(self):
        self.h_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)
        self.a_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)
        self.D_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2) 
        self.pD_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.9) 
    
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
    
    def define_discriminator(self):
        x_ = layers.Input(shape = (self.ydim,)) 
        z = layers.Input(shape = (self.adim,)) # Concentration Parameter

        if (self.fair_type == 'eo')|(self.fair_type == 'pp'):
            y = layers.Input(shape = (1,))
            cx = layers.Concatenate()([x_, z, y])
        else:
            cx = layers.Concatenate()([x_, z])
        
        x = layers.Dense(int(self.ldim), activation = None)(cx)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(int(self.ldim), activation = None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(1, activation = None)(x)
        
        if (self.fair_type == 'eo')|(self.fair_type == 'pp'):
            model = keras.Model(inputs = [x_, z, y], outputs = x, name = "discriminator")
        else:
            model = keras.Model(inputs = [x_, z], outputs = x, name = "discriminator")
        return model 

    def define_prediscriminator(self):
        y = layers.Input(shape = (self.ydim,)) 
        z = layers.Input(shape = (self.adim,)) # Concentration Parameter

        cx = layers.Concatenate()([y, z])
        
        x = layers.Dense(int(self.ldim), activation = None)(cx)
        x = layers.ReLU()(x)
        x = layers.Dense(int(self.ldim), activation = None)(x)
        x = layers.ReLU()(x)
        x = layers.Dense(1, activation = None)(x)
        
        model = keras.Model(inputs = [y,z], outputs = x, name = "pd")
        
        return model 
    
    
    @tf.function
    def train_model(self, X, A, Y):
        
        if self.fair_type == 'sp':
            neural, dYhatA, dYhatT = self.train_neural_netdist(X, A)
            main_loss = self.train_fair_classifier(X, A, Y)
        
        elif self.fair_type == 'vn':
            main_loss = self.train_fair_classifier(X, A, Y)
            neural = 0
            self.plam = 0 
            
        elif self.fair_type == 'eo':
            neural, dYhatA, dYhatT = self.train_neural_netdist_eo(X, A, Y)
            main_loss = self.train_fair_classifier_eo(X, A, Y)
        
        elif self.fair_type == 'pp':
            neural, dYhatA, dYhatT = self.train_neural_netdist_eo(X, A, Y)
            main_loss = self.train_fair_classifier_eo(X, A, Y)
        
        return main_loss, neural
    
    
    @tf.function
    def train_neural_netdist(self, X, A):
        for j in range(self.dround):
            with tf.GradientTape() as n:
                Yhat = self.h(X, training = True)
                
                dYhatA = self.da(self.D([Yhat, A], training = True))
                dYhatT = self.da(self.D([Yhat, tf.random.shuffle(A)], training = True))
                neural = self.neural_loss(dYhatA, dYhatT) 

            dgrad = n.gradient(neural, self.D.trainable_variables)
            if self.clip_norm == True:
                dgrad,_ = tf.clip_by_global_norm(dgrad, clip_norm=0.01)
            self.D_optim.apply_gradients(zip(dgrad, self.D.trainable_variables))
            
        return neural, tf.abs(tf.reduce_mean(dYhatA)), tf.abs(tf.reduce_mean(dYhatT))

    @tf.function
    def train_fair_classifier(self, X,A,Y):
        with tf.GradientTape() as t:

            YhatA = self.h(X, training = True)

            dYhatA = self.da(self.D([YhatA, A], training = True))
            dYhatT = self.da(self.D([YhatA, tf.random.shuffle(A)], training = True))

            main_loss = self.fit_loss(Y, YhatA)
            neural = self.neural_loss(dYhatA, dYhatT)
            total_loss = main_loss*(1 - self.plam) - neural * self.plam  # f will be tuned 

        grad = t.gradient(total_loss, self.h.trainable_variables)
        self.h_optim.apply_gradients(zip(grad, self.h.trainable_variables))
        return main_loss

    def generate_random_sensi(self, A):
        gen = tf.random.shuffle(A)
        return gen
    
    def get_weights(self, Y, A):
        if self.fair_type == 'eo':
#             w = self.da(self.pD([Y,A], training = False))
            w = tf.ones_like(A)*0.5
            return w/(1-w)
        
        elif self.fair_type == 'pp':
            weight = tf.reshape(tf.reduce_mean(A), (-1,1))
#             weight = 0.85
            w = 1/weight * A  + 1/(1-weight) * (1-A)
            return w
    
    @tf.function
    def pretrain_classifier(self, X, A, Y):

        with tf.GradientTape() as t: 
            ref = self.generate_random_sensi(A)
            dYhatA = self.da(self.pD([Y, A], training = True))
            dYhatT = self.da(self.pD([Y, ref], training = True))
            neural = self.neural_loss(dYhatA, dYhatT)
        
        dgrad = t.gradient(neural, self.pD.trainable_variables)
        self.pD_optim.apply_gradients(zip(dgrad, self.pD.trainable_variables))
        
        return neural
    
    def evaluate_pc(self, A, Y):
        ref = self.generate_random_sensi(A)
        dYhatA = self.da(self.pD([Y, A], training = False))
        dYhatT = self.da(self.pD([Y, ref], training = False))
        neural = self.neural_loss(dYhatA, dYhatT)

        return neural
    
    @tf.function
    def train_neural_netdist_eo(self, X,A,Y):
        for j in range(self.dround):
            weight = self.get_weights(Y,A)
            with tf.GradientTape() as n:
                
                ref = self.generate_random_sensi(A)
                YhatA = self.h(X, training = True)
                
                dYhatA = self.da(self.D([YhatA, A, Y], training = True))
                dYhatT = self.da(self.D([YhatA, ref, Y], training = True))
                neural = self.neural_wloss(dYhatA, dYhatT, weight) 

            dgrad = n.gradient(neural, self.D.trainable_variables)
            self.D_optim.apply_gradients(zip(dgrad, self.D.trainable_variables))
        return neural, tf.abs(tf.reduce_mean(dYhatA)), tf.abs(tf.reduce_mean(dYhatT))

    @tf.function
    def train_fair_classifier_eo(self, X,A,Y):
        
        weight = self.get_weights(Y,A)
        with tf.GradientTape() as t:

            YhatA = self.h(X, training = True)
            ref = self.generate_random_sensi(A)
            
            dYhatA = self.da(self.D([YhatA, A, Y], training = True))
            dYhatT = self.da(self.D([YhatA, ref, Y], training = True))

            main_loss = self.fit_loss(Y, YhatA)
            neural = self.neural_wloss(dYhatA, dYhatT, weight) 
            total_loss = main_loss*(1 - self.plam) - neural * self.plam 

        grad = t.gradient(total_loss, self.h.trainable_variables)
        self.h_optim.apply_gradients(zip(grad, self.h.trainable_variables))
        return main_loss

    @tf.function
    def train_prediscriminator(self, X, A, Y):
        with tf.GradientTape() as t: 
            YhatA = self.h(X, training = True)
            ref = self.generate_random_sensi(A)
            dYhatA = self.da(self.pD([YhatA, A], training = True))
            dYhatT = self.da(self.pD([YhatA, ref], training = True))
            neural = self.neural_loss(dYhatA, dYhatT)
        
        dgrad = t.gradient(neural, self.pD.trainable_variables)
        self.pD_optim.apply_gradients(zip(dgrad, self.pD.trainable_variables))
        
        return neural
    
    def evaluate(self, X):
        return self.h(X, training = False)
       

