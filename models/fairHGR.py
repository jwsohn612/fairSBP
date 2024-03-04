import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from tensorflow import keras

# The below code script is written by referring to the original work:   
# A Maximal Correlation Approach to Imposing Fairness in Machine Learning, Lee et al., 2022'.

class fairHGR(): 
    
    def __init__(self, loss_type, fair_type, out_act, learning_rate, dround, xdim, adim, plam, ldim):
        
        self.learning_rate = learning_rate 
        self.dround = dround 
        self.xdim = xdim
        self.adim = adim
        self.plam = plam
        self.ldim = ldim
        self.loss_type = loss_type
        self.fair_type = fair_type
        self.out_act = out_act 
            
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.ce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        
        self.define_optimizers()
        self.define_networks()
        
    def fit_loss(self, y, yhat):
        if self.loss_type == 'mae':
            return self.mae(y, yhat)
        if self.loss_type == 'mse':
            return self.mse(y, yhat)
        if self.loss_type == 'ce':
            return self.ce(y, yhat)
            
    def define_optimizers(self):
        self.f_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)
        self.T_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)
        self.e_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)
        self.g_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)
        if self.fair_type == "eo":
            self.e2_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)
            self.g2_optim = tf.keras.optimizers.SGD(learning_rate = self.learning_rate, momentum=0.2)
            
    def define_networks(self):
        self.f = self.define_classifier() # main decision model. 
        self.e = self.define_hgr_network(self.adim) # embedding A to a score
        self.g = self.define_hgr_network(1) # embedding f(x) to a score

        if self.fair_type == "eo":
            self.e2 = self.define_hgr_network(1) # for Y 
            self.g2 = self.define_hgr_network(1) 

    def define_hgr_network(self, input_dim):
        e = keras.Sequential([
            layers.InputLayer(input_dim,), 
            layers.Dense(int(self.ldim), activation=None), 
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(int(self.ldim), activation=None),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(1, activation=None)
        ])
        return e

    def define_classifier(self):
        f = keras.Sequential([
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
        return f

    @tf.function
    def get_empirical_cov(self, feat_map):
        size = feat_map.shape[0]
        return tf.matmul(feat_map,feat_map,transpose_a=True)/(size-1)

    @tf.function
    def evaluate_soft_HGR(self, e, g, X,A):
        size = X.shape[0]

        gfX = g(self.f(X, training=True), training=True) 
        gfX = gfX - tf.reduce_mean(gfX, axis=0, keepdims=True)
        eA = e(A, training = True) 
        eA = eA - tf.reduce_mean(eA, axis=0, keepdims=True)

        cov_gfX = self.get_empirical_cov(gfX)
        cov_eA = self.get_empirical_cov(eA)

        soft_hgr = tf.linalg.trace(tf.matmul(gfX, eA, transpose_b=True))/(size-1) - tf.linalg.trace(tf.multiply(cov_gfX, cov_eA))/2
        return soft_hgr

    @tf.function
    def train_model(self, X, A, Y):
        
        if self.fair_type == "sp":
            softhgr = self.train_maximization_step(X,A,Y)
            main_loss = self.train_minimization_step(X,A,Y)
            return main_loss, softhgr
        
        elif self.fair_type == "eo":
            softhgr1 = self.train_maximization_step_eo1(X,A,Y)
            softhgr2 = self.train_maximization_step_eo2(X,A,Y)
            main_loss = self.train_minimization_step_eo(X,A,Y)
            
            return main_loss, (softhgr1 - softhgr2)

    @tf.function
    def train_maximization_step(self, X,A,Y):
        for i in range(self.dround):
            with tf.GradientTape(persistent=True) as n:
                softhgr = self.evaluate_soft_HGR(self.e, self.g, X, A)
                C = -softhgr

            egrad = n.gradient(C, self.e.trainable_variables)
            ggrad = n.gradient(C, self.g.trainable_variables)

            self.e_optim.apply_gradients(zip(egrad, self.e.trainable_variables))
            self.g_optim.apply_gradients(zip(ggrad, self.g.trainable_variables))

        return softhgr

    @tf.function
    def train_minimization_step(self, X,A,Y):

        with tf.GradientTape(persistent=True) as n:

            C = self.fit_loss(Y, self.f(X, training = True))*(1-self.plam) + self.plam * self.evaluate_soft_HGR(self.e, self.g, X,A)

        fgrad = n.gradient(C, self.f.trainable_variables)

        self.f_optim.apply_gradients(zip(fgrad, self.f.trainable_variables))

        return C
    
    
    @tf.function
    def train_maximization_step_eo1(self, X,A,Y):
        for i in range(self.dround):
            with tf.GradientTape(persistent=True) as n:
                D = tf.multiply(A,Y)
                softhgr = self.evaluate_soft_HGR(self.e, self.g, X, D)
                C = -softhgr

            egrad = n.gradient(C, self.e.trainable_variables)
            ggrad = n.gradient(C, self.g.trainable_variables)

            self.e_optim.apply_gradients(zip(egrad, self.e.trainable_variables))
            self.g_optim.apply_gradients(zip(ggrad, self.g.trainable_variables))

        return softhgr
    
    @tf.function
    def train_maximization_step_eo2(self, X,A,Y):
        for i in range(self.dround):
            with tf.GradientTape(persistent=True) as n:
                softhgr = self.evaluate_soft_HGR(self.e2, self.g2, X, Y)
                C = -softhgr

            egrad = n.gradient(C, self.e2.trainable_variables)
            ggrad = n.gradient(C, self.g2.trainable_variables)

            self.e2_optim.apply_gradients(zip(egrad, self.e2.trainable_variables))
            self.g2_optim.apply_gradients(zip(ggrad, self.g2.trainable_variables))

        return softhgr

    
    @tf.function
    def train_minimization_step_eo(self, X,A,Y):

        with tf.GradientTape(persistent=True) as n:
            D = tf.multiply(A,Y)
            C = self.fit_loss(Y,self.f(X, training = True))*(1-self.plam) + self.plam *(self.evaluate_soft_HGR(self.e, self.g, X,D)-self.evaluate_soft_HGR(self.e2,self.g2,X,Y))

        fgrad = n.gradient(C, self.f.trainable_variables)

        self.f_optim.apply_gradients(zip(fgrad, self.f.trainable_variables))

        return C
    
    def evaluate(self, X):
        return self.f(X, training = False)
