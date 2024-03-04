import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

class data_loader():
    
    def __init__(self, data_dir, data_name, sensitive_attrs, batch_size = 200, epochs = 1000, vali_ratio=0.2, seed = 1):
        self.epochs = epochs
        self.batch_size = batch_size 
        self.data_name = data_name 
        self.seed = seed
        self.vali_ratio = 0.2 
        self.sensitive_attrs = sensitive_attrs
        self.data_dir = data_dir 
        
    def get_data(self):
        
        if self.data_name == "adult":
            data_tuple = self.import_adult()
        if self.data_name == "crime":
            data_tuple = self.import_crime()
        if self.data_name == "credit":
            data_tuple = self.import_credit()
        if self.data_name == "law":
            data_tuple = self.import_law()
        if self.data_name == "employment":
            data_tuple = self.import_employment()
        return data_tuple
    
    def import_adult(self):
        
        sensitive_attrs = self.sensitive_attrs
        
        data =  pd.read_csv(self.data_dir + 'adult.data', sep=",", header = None)
        scaler = MinMaxScaler(feature_range=(-1,1))

        data.columns = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label']

        data = data.drop(["education",'fnlwgt'],axis=1) # education : category, fnlwgt : continuous
        
        conti_vars = ['education_num','capital_gain','capital_loss','hours_per_week']
        cate_vars = ['workclass','marital_status','occupation','relationship','native_country']
        label_var = 'label'
            
        # Preprocessing # 
        data = data.assign(label = (data['label']==' >50K').astype(int), sex = (data['sex']==' Male').astype(int))
        data = data.assign(race = ["W"  if x==' White' else "NW"  for x in data['race']])
        
        conti_vars_training = [x for x in conti_vars if x not in sensitive_attrs]
        cate_vars_training = [x for x in cate_vars if x not in sensitive_attrs]

        if 'age' in sensitive_attrs:
            data_sensi_cate = [x for x in ['race','sex'] if x in sensitive_attrs]
            sensitive_df = pd.DataFrame(scaler.fit_transform(data[['age']]))

            if len(data_sensi_cate) > 0:
                discrete_df = pd.get_dummies(data[data_sensi_cate], drop_first=True)        
                sensitive_df = pd.concat([sensitive_df, discrete_df], axis=1)

        else:
            sensitive_df = pd.get_dummies(data[sensitive_attrs], drop_first=True)
                
        XC = scaler.fit_transform(data[conti_vars_training])
        XD = np.array(pd.get_dummies(data[cate_vars_training], drop_first=True))
        X = np.concatenate((XC,XD), axis=1)
        Y = np.array(data[[label_var]])
        A = np.array(sensitive_df)
        
        np.random.seed(self.seed)
        idx = np.random.choice([x for x in range(data.shape[0])], size = int(data.shape[0]*self.vali_ratio), replace=False)
        train_idx = list(set([x for x in range(data.shape[0])]) - set(idx))
        
        return self.get_data_output(X,A,Y, train_idx, idx)

    def import_employment(self):
        
        sensitive_attrs = self.sensitive_attrs
        
        data =  pd.read_csv(self.data_dir + 'ACSEmployment_CA.csv', sep=",")
        scaler = MinMaxScaler(feature_range=(-1,1))

        cate_vars = ['SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC','NATIVITY', 'DEAR', 'DEYE', 'DREM']
        label_var = 'ESR'
            
        # Preprocessing # 
        data = data.assign(SEX = [1  if x==1 else 0 for x in data['SEX']]) # 1:male  - 2:femal
        
        cate_vars_training = [x for x in cate_vars if x not in sensitive_attrs]
                
       
        if 'AGEP' in sensitive_attrs:
            data_sensi_cate = [x for x in ['SEX'] if x in sensitive_attrs]
            sensitive_df = pd.DataFrame(scaler.fit_transform(data[['AGEP']]))

            if len(data_sensi_cate) > 0:
                discrete_df = pd.get_dummies(data[data_sensi_cate], drop_first=True)        
                sensitive_df = pd.concat([sensitive_df, discrete_df], axis=1)
            else:
                sensitive_df = pd.get_dummies(data[sensitive_attrs], drop_first=True)
                conti_vars_training.append('AGEP')
                XC = np.array(data[conti_vars_training])
        
        
        X = np.array(pd.get_dummies(data[cate_vars_training].astype(str), drop_first=True))
        Y = np.array(data[[label_var]])
        A = np.array(sensitive_df)
        
        np.random.seed(self.seed)
        idx = np.random.choice([x for x in range(data.shape[0])], size = int(data.shape[0]*self.vali_ratio), replace=False)
        train_idx = list(set([x for x in range(data.shape[0])]) - set(idx))
        
        return self.get_data_output(X,A,Y, train_idx, idx)
    
    def import_crime(self):
        
        sensitive_attrs = self.sensitive_attrs
            
        data =  pd.read_csv(self.data_dir + 'crime_normalized.csv') 
        data.fillna(data.mean(numeric_only=True).round(1), inplace=True)
        data = data[[x for x in data.columns if x not in ['Unnamed: 0', 'state','county','community','communityname','fold']]]
        
        np.random.seed(self.seed)
        idx = np.random.choice([x for x in range(data.shape[0])], size = int(data.shape[0]*self.vali_ratio), replace=False)
        train_idx = list(set([x for x in range(data.shape[0])]) - set(idx))
    
        A = np.array(data[sensitive_attrs])
        Y = np.array(data[['ViolentCrimesPerPop']])
        sensitive_attrs.append('ViolentCrimesPerPop')
        X = np.array(data[[x for x in data.columns if x not in sensitive_attrs]])
        
        return self.get_data_output(X,A,Y, train_idx, idx)
    
    def import_credit(self):

        sensitive_attrs = self.sensitive_attrs # AGE, SEX

        data =  pd.read_csv(self.data_dir + 'UCI_Credit_Card.csv')
        scaler = MinMaxScaler(feature_range=(-1,1))

        data_columns = [x for x in data.columns]
        data[['SEX']] = data[['SEX']].replace({1:'m', 2:'f'})
        
        conti_vars = [x for x in data_columns if x not in ['ID','SEX','EDUCATION','MARRIAGE','AGE','default.payment.next.month']]
        cate_vars = ['EDUCATION','MARRIAGE']
        label_var = 'default.payment.next.month'

        conti_vars_training = [x for x in conti_vars if x not in sensitive_attrs]
        cate_vars_training = [x for x in cate_vars if x not in sensitive_attrs]

        
        if 'AGE' in sensitive_attrs:
            data_sensi_cate = [x for x in ['SEX'] if x in sensitive_attrs]
            sensitive_df = pd.DataFrame(scaler.fit_transform(data[['AGE']]))

            if len(data_sensi_cate) > 0:
                discrete_df = pd.get_dummies(data[data_sensi_cate], drop_first=True)        
                sensitive_df = pd.concat([sensitive_df, discrete_df],axis=1)
        else:
            sensitive_df = pd.get_dummies(data[sensitive_attrs], drop_first=True)

      
        XC = scaler.fit_transform(data[conti_vars_training])
        XD = np.array(pd.get_dummies(data[cate_vars_training], drop_first=True))
        X = np.concatenate((XC,XD), axis=1)
        Y = np.array(data[[label_var]])
        A = np.array(sensitive_df)

        np.random.seed(self.seed)
        idx = np.random.choice([x for x in range(data.shape[0])], size = int(data.shape[0]*self.vali_ratio), replace=False)
        train_idx = list(set([x for x in range(data.shape[0])]) - set(idx))

        return self.get_data_output(X,A,Y, train_idx, idx)

    def import_law(self):

        sensitive_attrs = self.sensitive_attrs

        data =  pd.read_csv(self.data_dir + 'lawschool.csv')
        data = data.loc[lambda x:x['MissingRace']==0].dropna()
        
        scaler = MinMaxScaler(feature_range=(-1,1))

        conti_vars = ['LSAT','GPA']
        cate_vars = ['college','Year']
        label_var = 'admit'

        # Preprocessing # 
        data = data.assign(Race = [1  if x=='White' else 0  for x in data['Race']])


        XC = scaler.fit_transform(data[conti_vars])
        XD = np.array(pd.get_dummies(data[cate_vars], drop_first=True))
        X = np.concatenate((XC,XD), axis=1)
        Y = np.array(data[[label_var]])
        A = np.array(data[sensitive_attrs])

        np.random.seed(self.seed)
        idx = np.random.choice([x for x in range(data.shape[0])], size = int(data.shape[0]*self.vali_ratio), replace=False)
        train_idx = list(set([x for x in range(data.shape[0])]) - set(idx))

        return self.get_data_output(X,A,Y, train_idx, idx)
        
    def get_data_output(self, X, A, Y, train_idx, idx, softmax=False):
        
        TrainA = tf.cast(A[train_idx,:], dtype=tf.float32)
        TrainY = tf.cast(Y[train_idx,:], dtype=tf.float32)
        TrainX = tf.cast(X[train_idx,:], dtype=tf.float32)
        
        
        ValiA = tf.cast(A[idx,:], dtype=tf.float32)
        ValiY = tf.cast(Y[idx,:], dtype=tf.float32)
        ValiX = tf.cast(X[idx,:], dtype=tf.float32)

        if softmax==True:
            # One-hot encode the binary labels
            TrainY = tf.keras.utils.to_categorical(TrainY)
            ValiY = tf.keras.utils.to_categorical(ValiY)
   
        xdim = TrainX.shape[1]
        adim = TrainA.shape[1]
        ydim = TrainY.shape[1]
        

        buffer_size = TrainA.shape[0]
        batch_size = self.batch_size
        epochs = self.epochs
        
        train_ds = (
          tf.data.Dataset.from_tensor_slices((TrainX, TrainA, TrainY))
          .shuffle(buffer_size, reshuffle_each_iteration = True)
          .repeat(epochs)
          .batch(batch_size, drop_remainder=True)

        )
        
        return (train_ds, (ValiX, ValiA, ValiY), (xdim,adim,ydim))

