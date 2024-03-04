import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from tensorflow import keras



class fairCONR(): 
    
    def __init__(self, loss_type, fair_type, out_act, learning_rate, dround, xdim, adim, plam, ldim, syn_type):
        
        self.learning_rate = learning_rate 
        self.dround = dround 
        self.xdim = xdim
        self.adim = adim
        self.ydim = 1
        self.plam = plam
        self.ldim = ldim
        self.gamma = 1
        self.syn_type = syn_type # binary
        
        self.zdim = 2
        
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
        
        # Define Conditional Sampler 
        self.d = self.define_GAN_d()
        if syn_type == 2:
            self.g = self.define_GAN_g()
        elif syn_type == 3:
            self.g = self.define_GAN_g2()
        
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
        
        self.g_optim = tf.keras.optimizers.Adam(learning_rate = 0.0002)
        self.d_optim = tf.keras.optimizers.Adam(learning_rate = 0.0002)
        
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
    
    
    def define_GAN_d(self):
        z = layers.Input(shape = (self.adim,)) 
        x_ = layers.Input(shape = (self.ydim,)) 
        
        cx = layers.Concatenate()([x_, z])
        x = layers.Dense(int(self.ldim), activation = None)(cx)
        x = layers.ReLU()(x)
        x = layers.Dense(int(self.ldim), activation = None)(x)
        x = layers.ReLU()(x)
        x = layers.Dense(1, activation = 'sigmoid')(x)
        
        model = keras.Model(inputs = [z, x_], outputs = x, name = "discriminator")
        return model
    
    def define_GAN_g(self):
        z = layers.Input(shape = (self.zdim,))  # NOISE
        x_ = layers.Input(shape = (self.ydim,))  # OUTCOME
        
        cx = layers.Concatenate()([x_, z])
        x = layers.Dense(int(self.ldim), activation = None)(cx)
        x = layers.ReLU()(x)
        x = layers.Dense(int(self.ldim), activation = None)(x)
        x = layers.ReLU()(x)
        a1 = layers.Dense(1, activation = None)(x)
        a0 = layers.Dense(1, activation = 'sigmoid')(x)
        
        a = layers.Concatenate()([a1, a0])
        
        model = keras.Model(inputs = [z, x_], outputs = a, name = "generator")
        return model
    
    def define_GAN_g2(self):
        z = layers.Input(shape = (self.zdim,))  # NOISE
        x_ = layers.Input(shape = (self.ydim,))  # OUTCOME
        
        cx = layers.Concatenate()([x_, z])
        x = layers.Dense(int(self.ldim), activation = None)(cx)
        x = layers.ReLU()(x)
        x = layers.Dense(int(self.ldim), activation = None)(x)
        x = layers.ReLU()(x)
        a = layers.Dense(1, activation = None)(x)
         
        model = keras.Model(inputs = [z, x_], outputs = a, name = "generator")
        return model
    
    def generate_random_noise(self, size):
        return tf.random.uniform((size, self.zdim))
    
    @tf.function
    def pretrain_conditional_sampler(self, A, Y):
        
        z = self.generate_random_noise(A.shape[0])
        with tf.GradientTape(persistent=True) as d_tape:
            Ag = self.g([z,Y],training = True)
        
            l_r = self.d([A,Y],training = True) 
            l_g = self.d([Ag,Y],training = True)
            
            d_loss = self.neural_loss(l_r, l_g)
            g_loss = -d_loss
            
        d_grad = d_tape.gradient(d_loss, self.d.trainable_variables)
        g_grad = d_tape.gradient(g_loss, self.g.trainable_variables)
        
        self.d_optim.apply_gradients(zip(d_grad, self.d.trainable_variables))
        self.g_optim.apply_gradients(zip(g_grad, self.g.trainable_variables))
        
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

    
    @tf.function
    def train_model(self, X, A, Y):
        
        if self.fair_type == 'eo':
            neural, dYhatA, dYhatT = self.train_neural_netdist_eo(X, A, Y)
            main_loss = self.train_fair_classifier_eo(X, A, Y)
                
        return main_loss, neural
    

    def get_binary_probs(self, A, Y):
        '''
        For binary A
        '''
        pA1_Y1 = tf.reduce_mean(A[tf.reshape(Y,(-1))==1])
        pA1_Y0 = tf.reduce_mean(A[tf.reshape(Y,(-1))==0])
        self.pA_Y = tf.cast(np.array([pA1_Y0, pA1_Y1]),dtype=tf.float32)
        
    
    def generate_random_sensi_binary_case(self, A,Y):
        Y_ = tf.reshape(tf.cast(Y, dtype=tf.int32), (-1,))
        pA = tf.gather(self.pA_Y, Y_)
        u = tf.random.uniform((A.shape[0],))
        pA_ = tf.reshape(pA, (-1,))
        gen = tf.cast(pA_ > u, dtype=tf.float32)
        return tf.reshape(gen, A.shape)
    
    def model_based_sensi(self, Y):
        z = self.generate_random_noise(Y.shape[0])
        ag = self.g([z,Y])
        a_con = tf.reshape(ag[:,0], (-1,1))
        a_dis = tf.reshape(ag[:,1], (-1,1))
        a_dis = tf.where(a_dis>=0.5, tf.ones_like(a_dis), a_dis)
        a_dis = tf.where(a_dis<0.5, tf.zeros_like(a_dis), a_dis)
        return tf.concat([a_con,a_dis],axis=1)
    
    def model_based_sensi2(self, Y):
        z = self.generate_random_noise(Y.shape[0])
        ag = self.g([z,Y])
        a_con = tf.reshape(ag, (-1,1))
        return a_con
    
    @tf.function
    def generate_random_sensi(self, A, Y):
        if self.syn_type == 1:
            return self.generate_random_sensi_binary_case(A,Y)
        elif self.syn_type == 2: 
            return self.model_based_sensi(Y)
        elif self.syn_type == 3:
            return self.model_based_sensi2(Y)
    
    def get_weights(self, Y, A):
        if self.fair_type == 'eo':
            w = self.da(self.pD([Y,A], training = False))
            #w = tf.ones_like(A)*0.5
            return w/(1-w)
        
        elif self.fair_type == 'pp':
            weight = tf.reshape(tf.reduce_mean(A), (-1,1))
#             weight = 0.85
            w = 1/weight * A  + 1/(1-weight) * (1-A)
            return w
    
    @tf.function
    def train_neural_netdist_eo(self, X,A,Y):
        for j in range(self.dround):
            weight = self.get_weights(Y,A)
            with tf.GradientTape() as n:
                
                ref = self.generate_random_sensi(A, Y)
                
                YhatA = self.h(X, training = True)
                
                dYhatA = self.da(self.D([YhatA, A, Y], training = True))
                dYhatT = self.da(self.D([YhatA, ref, Y], training = True))
                neural = self.neural_loss(dYhatA, dYhatT) 

            dgrad = n.gradient(neural, self.D.trainable_variables)
            self.D_optim.apply_gradients(zip(dgrad, self.D.trainable_variables))
        return neural, tf.abs(tf.reduce_mean(dYhatA)), tf.abs(tf.reduce_mean(dYhatT))
    
    @tf.function
    def get_cov(self, x, y):
#         if self.syn_type == 1:
        mean_x = tf.reduce_mean(x)
        mean_y = tf.reduce_mean(y, axis=0)

        centered_x = x - mean_x
        centered_y = y - mean_y

        covariance = tf.reduce_mean(centered_x * centered_y)
        
        return covariance

    @tf.function
    def train_fair_classifier_eo(self, X,A,Y):
        
        weight = self.get_weights(Y,A)
        with tf.GradientTape() as t:

            YhatA = self.h(X, training = True)
            ref = self.generate_random_sensi(A, Y)
            
            dYhatA = self.da(self.D([YhatA, A, Y], training = True))
            dYhatT = self.da(self.D([YhatA, ref, Y], training = True))

            main_loss = self.fit_loss(Y, YhatA)
            neural = self.neural_loss(dYhatA, dYhatT) 
            
            cov1 = self.get_cov(YhatA, A)
            cov2 = self.get_cov(YhatA, ref)
            
            total_loss = main_loss*(1 - self.plam) - neural * self.plam +  self.plam*self.gamma*(cov1 - cov2)**2

        grad = t.gradient(total_loss, self.h.trainable_variables)
        self.h_optim.apply_gradients(zip(grad, self.h.trainable_variables))
        return main_loss

    def evaluate(self, X):
        return self.h(X, training = False)
       

