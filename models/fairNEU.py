import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from tensorflow import keras

# The below code script is written by referring to the original work:   
# Fairness via Represenstation Neutralization, 'Du et al., 2021'.

class fairNEU(): 
    
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
            
        self.mdim = 32
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.ce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
       
        self.lam_list = tf.cast(np.random.choice(a=[0.6,0.7,0.8,0.9], size=10000),dtype=tf.float32)
        self.define_optimizers()
        self.define_networks()


    def da(self, x):
        return tf.math.sigmoid(x)

    def neural_loss(self, x1,x2):
        real_loss = self.ce(tf.ones_like(x1), x1)
        fake_loss = self.ce(tf.zeros_like(x2), x2)
        D_loss = real_loss + fake_loss
        return  D_loss

    def fit_loss(self, y, yhat):
        if self.loss_type == 'mae':
            return self.mae(y, yhat)
        if self.loss_type == 'mse':
            return self.mse(y, yhat)
        if self.loss_type == 'ce':
            return self.ce(y, yhat)
            
    def define_optimizers(self):
        self.g_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)
        self.T_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)

    def define_networks(self):
        self.g = keras.Sequential([
            layers.InputLayer(self.xdim,), 
            layers.Dense(self.ldim, activation=None), 
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        
        self.T = keras.Sequential([
            layers.InputLayer(self.ldim,), 
            layers.Dense(self.ldim, activation=None),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(self.ldim, activation=None),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(1, activation=self.out_act)
        ])
    
    @tf.function
    def forward(self, X):
        return self.T(self.g(X, training = True), training = True)
    
    @tf.function
    def pretrain_classifier(self, X, A, Y):
        with tf.GradientTape(persistent=True) as t: 
            Yhat = self.forward(X)
            main_loss = self.fit_loss(Y, Yhat)
            
        ggrad = t.gradient(main_loss, self.g.trainable_variables)
        tgrad = t.gradient(main_loss, self.T.trainable_variables)
    
        self.g_optim.apply_gradients(zip(ggrad, self.g.trainable_variables))
        self.T_optim.apply_gradients(zip(tgrad, self.T.trainable_variables))
        
        return main_loss
    
    @tf.function
    def train_classifier_head(self, Y1, Z1, Y1_, Z1_, Y0, Z0, Y0_, Z0_):
        
        lam1 = tf.reshape(tf.random.shuffle(self.lam_list)[:Z1.shape[0]],(-1,1))
        lam0 = tf.reshape(tf.random.shuffle(self.lam_list)[:Z0.shape[0]],(-1,1))
        
        with tf.GradientTape(persistent=True) as t: 

            P1 = self.T(Z1, training = True)
            P1_ = self.T(Z1_, training = True)
            
            P0 = self.T(Z0, training = True)
            P0_ = self.T(Z0_, training = True)
            
            z1 = 0.5*Z1 + 0.5*Z1_
            z0 = 0.5*Z0 + 0.5*Z0_
            
            sz1 = lam1*Z1 + (1-lam1)*Z1_
            sz0 = lam0*Z0 + (1-lam0)*Z0_
            
            p1 = 0.5*P1 + 0.5*P1_
            p0 = 0.5*P0 + 0.5*P0_
            
            z = tf.concat([z1,z0], axis=0)
            sz = tf.concat([sz1,sz0], axis=0)
            p = tf.concat([p1,p0], axis=0)
            
            Lmse = self.mse(p, self.T(z, training = True))
            Lmae = self.mae(self.T(sz, training = True), self.T(z, training = True))
            
            main_loss = Lmse *(1-self.plam) + self.plam * Lmae

        tgrad = t.gradient(main_loss, self.T.trainable_variables)
        self.T_optim.apply_gradients(zip(tgrad, self.T.trainable_variables))
        
        return main_loss
    
    def get_random_idx(self, Y, label):
        idx = tf.where(tf.equal(Y, label))[:, 0]
        return tf.random.shuffle(idx)
    
    @tf.function
    def make_training_datasets(self, X, Y):
        
        size = X.shape[0]
        Z = self.g(X, training = True)

        Y1 = Y[Y==1]
        Y0 = Y[Y==0]
        Z1 = tf.gather(Z, indices=np.where(Y.numpy()==1)[0])
        Z0 = tf.gather(Z, indices=np.where(Y.numpy()==0)[0])
    
        idx1 = self.get_random_idx(Y, 1) 
        idx0 = self.get_random_idx(Y, 0)

        Y1_ = tf.gather(Y, indices=idx1)
        Y0_ = tf.gather(Y, indices=idx0)
        Z1_ = tf.gather(Z, indices=idx1)
        Z0_ = tf.gather(Z, indices=idx0)
        
        return Y1, Z1, Y1_, Z1_, Y0, Z0, Y0_, Z0_
    
    @tf.function
    def train_model(self, X, A, Y):
        
        # _ = self.train_classifier(X,Y)
        Y1, Z1, Y1_, Z1_, Y0, Z0, Y0_, Z0_ = self.make_training_datasets(X, Y)
        main_loss = self.train_classifier_head(Y1, Z1, Y1_, Z1_, Y0, Z0, Y0_, Z0_)
        return main_loss, tf.zeros(1)
    
    def evaluate(self, X):
        return self.T(self.g(X, training = False), training = False)