import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from data_loader import data_loader
from fairSBP import fairSBP
from fairKDE import fairKDE
from fairHGR import fairHGR
from fairNEU import fairNEU
from fairCONR import fairCONR
from utils import basic_metrics, evaluation_metrics, evaluation_metrics_cont, evaluation_metrics_contout
from tqdm import tqdm
import os
from scipy.stats import kde
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default = 1000, help = "number of training epochs", type = int)
parser.add_argument("--plambda", default = 0, help = "lambda", type = float)
parser.add_argument("--dround", default = 1, help = "T'", type = int)
parser.add_argument("--data-name", help = "name of data to be imported", default = None)
parser.add_argument("--learning_rate", default = 0.005, type = float)
parser.add_argument("--save-name", default = None)
parser.add_argument("--fair-type", help = "choose either es or eo", default = None)
parser.add_argument("--seed", default = 20220702, type=int)
parser.add_argument("--method", help = "method to be experimented", default = None)

args = parser.parse_args()

# first_col_cont = False # Scenario I
first_col_cont = True # Scenario II
binarize = False

parent_dir = "/home/sohn24/Desktop/fairness/simulation/" + args.data_name + "_" + args.fair_type + '/'
data_dir = "/home/sohn24/Desktop/fairness/simulation/datasets/"

data_name = args.data_name
fair_type = args.fair_type
seed = args.seed

save_unit = 100
save_unitc = 100

save_path =  os.path.join(parent_dir, args.save_name+ "/")
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Directory '{save_path}' created successfully!")
else:
    print(f"Directory '{save_path}' already exists!")
    
def get_gpu_name():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=name', '--format=csv,noheader'
        ], universal_newlines=True)
    gpu_names = result.strip().split('\n')
    return gpu_names

gpu_names = get_gpu_name()

if gpu_names:
    print(f"Found {len(gpu_names)} GPU(s): {', '.join(gpu_names)}")
else:
    print("No GPU found.")

plam = args.plambda
seed = args.seed
dround = args.dround
method = args.method
learning_rate = args.learning_rate

if data_name == 'adult':
    if method == "fairPRED":
        epochs = 50
    else:
        epochs = 600
        
    batch_size = 300
    ldim = 64 # 256 for Scenario2
    loss_type = 'ce'
    out_act = 'sigmoid'
    if first_col_cont == False:
        sensitive_attrs = ['race']
    else:
        sensitive_attrs = ['age','race']
    conepochs0 = 20000
    
elif data_name == 'employment':
    if method == "fairPRED":
        epochs = 50
    else:
        epochs = 50
        
    batch_size = 300
    ldim = 64 # 256 for Scenario2
    loss_type = 'ce'
    out_act = 'sigmoid'
    if first_col_cont == False:
        sensitive_attrs = ['SEX']
    else:
        sensitive_attrs = ['AGEP','SEX']
    conepochs0 = 20000
        
elif data_name == 'law':
    if method == "fairPRED":
        epochs = 50
    else:
        epochs = 600
        
    batch_size = 300
    ldim = 64
    loss_type = 'ce'
    out_act = 'sigmoid'
    if first_col_cont == False:
        sensitive_attrs = ['Race']
    else:
        sensitive_attrs = None

elif data_name == 'credit':
    if method == "fairPRED":
        epochs = 50
    else:
        epochs = 600
        
    batch_size = 300
    ldim = 64
    loss_type = 'ce'
    out_act = 'sigmoid'
    if first_col_cont == False:
        sensitive_attrs = ['SEX']
    else:
        sensitive_attrs = ['AGE','SEX']
    conepochs0 = 20000
    
elif data_name == 'crime':
    epochs = 1000
    batch_size = 300
    ldim = 16
    loss_type = 'mae'
    out_act = None 
    sensitive_attrs = ['racepctblack']
    conepochs0 = 20000
    
dl = data_loader(data_dir=data_dir, data_name=data_name, sensitive_attrs = sensitive_attrs, batch_size=batch_size, epochs=epochs, seed=seed)
data = dl.get_data(binarize_conti_sen=binarize)

train_ds = data[0]
(VX,VA,VY) = data[1]

np.mean(VY)

if method == "fairSBP":
    model = fairSBP(loss_type = loss_type, fair_type = fair_type, out_act=out_act,learning_rate=learning_rate, dround = dround, xdim = data[2][0], adim = data[2][1], plam = plam, ldim = ldim)

elif method == "fairCONR":
    if first_col_cont == False:
        syn_type = 1
    else:
        syn_type = 2
    if data_name == 'crime':
        syn_type = 3
        
    model = fairCONR(loss_type = loss_type, fair_type = fair_type, out_act=out_act,learning_rate=learning_rate, dround = dround, xdim = data[2][0], adim = data[2][1], plam = plam, ldim = ldim, syn_type=syn_type)
    
elif method == "fairHGR":
    model = fairHGR(loss_type = loss_type, fair_type = fair_type, out_act=out_act,learning_rate=learning_rate, dround = dround, xdim = data[2][0], adim = data[2][1], plam = plam, ldim = ldim)
    
elif method == "fairKDE":
    model = fairKDE(loss_type = loss_type, fair_type = fair_type, out_act=out_act,learning_rate=learning_rate, dround = dround, xdim = data[2][0], adim = data[2][1], plam = plam, ldim = ldim)   
    
elif method == "fairNEU":
    model = fairNEU(loss_type = loss_type, fair_type = fair_type, out_act=out_act,learning_rate=learning_rate, dround = dround, xdim = data[2][0], adim = data[2][1], plam = plam, ldim = ldim) 
    
var_loc0_list = []
var_loc1_list = []
var_loc2_list = []

basic_info_list = []

if (method == 'fairCONR')&(first_col_cont==False):
    model.get_binary_probs(VA,VY)
    
# Set up early stopping
best_val_loss = float('inf')
if data_name != 'crime':
    early_stopping_patience = 100
else: 
    early_stopping_patience = 100
    
if (method == "fairSBP"):
    early_stopping_patience = 100

early_stopping_counter = 0

t1 = datetime.now()
if (method == "fairNEU")|((method == 'fairSBP')&(fair_type=='eo')):

    for i, train_set in tqdm(enumerate(train_ds)):
        X = train_set[0]
        A = train_set[1]
        Y = train_set[2]

        loss = model.pretrain_classifier(X,A,Y)
        
        if (method == 'fairSBP'): 
            val_loss = model.evaluate_pc(VA,VY)
        else:
            val_loss = model.fit_loss(VY, model.evaluate(VX))
            
        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print('Early stopping!')
                break
                
conepochs = 0
if (method=='fairCONR')&(first_col_cont==True):
    for i, train_A in tqdm(enumerate(train_ds)):
        conepochs += 1
        X = train_A[0]
        A = train_A[1]
        Y = train_A[2]
        
        loss = model.pretrain_conditional_sampler(A,Y)
        
        if conepochs >= conepochs0:
            break
            
for i, train_set in tqdm(enumerate(train_ds)):
    
    X = train_set[0]
    A = train_set[1]
    Y = train_set[2]
    
    _, _ = model.train_model(X,A,Y)
    
    if i % save_unit == 0:
        
        ypred = model.evaluate(VX)
        train_loss = model.fit_loss(Y, model.evaluate(X))
        test_loss = model.fit_loss(VY, ypred)
        
        if data_name != 'crime':
            if (first_col_cont == False)|(binarize == True):
                out_df0 = evaluation_metrics(i, VA, VY, ypred, var_loc=0)
            else:
                out_df0 = evaluation_metrics_cont(i, VA, VY, ypred, var_loc=0)
        else:
            out_df0 = evaluation_metrics_contout(i, VA, VY, ypred, var_loc=0, util=test_loss.numpy())

        var_loc0_list.append(out_df0)
        
        if A.shape[1]==2:
            out_df1 = evaluation_metrics(i, VA, VY, ypred, var_loc=1)
            var_loc1_list.append(out_df1)
        
        t2 = datetime.now()
        cp_time = t2-t1

        basic_df = basic_metrics(i, plam, train_loss.numpy(), test_loss.numpy(), cp_time, gpu_names[0])
        basic_info_list.append(basic_df)

pd.concat(basic_info_list,axis=0).to_csv(save_path + "basic_{}_{}.csv".format(int(plam*100),seed))
pd.concat(var_loc0_list,axis=0).to_csv(save_path + "varloc0_{}_{}.csv".format(int(plam*100),seed))
if A.shape[1]==2:
    pd.concat(var_loc1_list,axis=0).to_csv(save_path + "varloc1_{}_{}.csv".format(int(plam*100),seed))

