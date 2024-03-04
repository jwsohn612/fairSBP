import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler



class data_loader():
    
    def __init__(self, data_dir, data_name, sensitive_attrs, batch_size = 200, epochs = 1000, vali_ratio=0.2, binarize=False, seed = 1):
        self.epochs = epochs
        self.batch_size = batch_size 
        self.data_name = data_name 
        self.seed = seed
        self.vali_ratio = 0.2 
        self.sensitive_attrs = sensitive_attrs
        self.data_dir = data_dir 
        self.binarize_conti_sen = binarize
        
    def get_data(self, binarize_conti_sen=False):
        
        if self.data_name == "adult":
            data_tuple = self.import_adult(binarize_conti_sen)
        if self.data_name == "crime":
            data_tuple = self.import_crime(binarize_conti_sen)
        if self.data_name == "compas":
            data_tuple = self.import_compas(binarize_conti_sen)
        if self.data_name == "credit":
            data_tuple = self.import_credit(binarize_conti_sen)
        if self.data_name == "law":
            data_tuple = self.import_law(binarize_conti_sen)
        if self.data_name == "insurance":
            data_tuple = self.import_insurance(binarize_conti_sen)
        if self.data_name == "employment":
            data_tuple = self.import_employment(binarize_conti_sen)
        return data_tuple
    
    def import_adult(self, binarize_conti_sen=False):
        '''
        Senstivie attributes list : age, race, sex
        binary output
        Output = [train_ds, (ValiA, ValiY, ValiX), (xdim,adim)]
        '''
        
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

        if binarize_conti_sen==False:
            if 'age' in sensitive_attrs:
                data_sensi_cate = [x for x in ['race','sex'] if x in sensitive_attrs]
                sensitive_df = pd.DataFrame(scaler.fit_transform(data[['age']]))

                if len(data_sensi_cate) > 0:
                    discrete_df = pd.get_dummies(data[data_sensi_cate], drop_first=True)        
                    sensitive_df = pd.concat([sensitive_df, discrete_df], axis=1)

            else:
                sensitive_df = pd.get_dummies(data[sensitive_attrs], drop_first=True)
        elif binarize_conti_sen==True:
            if 'age' in sensitive_attrs:
                avg_age = np.mean(data['age'])
                data = data.assign(age = [1 if x > avg_age else 0 for x in data['age']])
                sensitive_df = pd.get_dummies(data[sensitive_attrs], drop_first=True)
                
        #XC = np.array(data[conti_vars_training])
        XC = scaler.fit_transform(data[conti_vars_training])
        XD = np.array(pd.get_dummies(data[cate_vars_training], drop_first=True))
        X = np.concatenate((XC,XD), axis=1)
        Y = np.array(data[[label_var]])
        A = np.array(sensitive_df)
        
        np.random.seed(self.seed)
        idx = np.random.choice([x for x in range(data.shape[0])], size = int(data.shape[0]*self.vali_ratio), replace=False)
        train_idx = list(set([x for x in range(data.shape[0])]) - set(idx))
        
        return self.get_data_output(X,A,Y, train_idx, idx)

    
    def import_insurance(self, binarize_conti_sen=False):
        '''
        Senstivie attributes list : age, sex
        continuous output
        Output = [train_ds, (ValiA, ValiY, ValiX), (xdim,adim)]
        '''
        
        sensitive_attrs = self.sensitive_attrs
        
        data =  pd.read_csv(self.data_dir + 'insurance.csv', sep=",")
        scaler = MinMaxScaler(feature_range=(-1,1))

        conti_vars = ['bmi','children']
        cate_vars = ['smoker', 'region']
        label_var = 'charges'
            
        # Preprocessing # 
        data = data.assign(sex = [1  if x==' male' else 0 for x in data['sex']])
        
        conti_vars_training = [x for x in conti_vars if x not in sensitive_attrs]
        cate_vars_training = [x for x in cate_vars if x not in sensitive_attrs]

        if 'age' in sensitive_attrs:
    
            data_sensi_cate = [x for x in ['sex'] if x in sensitive_attrs]
            sensitive_df = pd.DataFrame(scaler.fit_transform(data[['age']]))

            if len(data_sensi_cate) > 0:
                discrete_df = pd.get_dummies(data[data_sensi_cate], drop_first=True)        
                sensitive_df = pd.concat([sensitive_df, discrete_df], axis=1)

        else:
            sensitive_df = pd.get_dummies(data[sensitive_attrs], drop_first=True)
            #conti_vars_training.append('age')
            
        #XC = np.array(data[conti_vars_training])
        XC = scaler.fit_transform(data[conti_vars_training])
        XD = np.array(pd.get_dummies(data[cate_vars_training], drop_first=True))
        X = np.concatenate((XC,XD), axis=1)
        Y = np.array(data[[label_var]])
        A = np.array(sensitive_df)
        
        np.random.seed(self.seed)
        idx = np.random.choice([x for x in range(data.shape[0])], size = int(data.shape[0]*self.vali_ratio), replace=False)
        train_idx = list(set([x for x in range(data.shape[0])]) - set(idx))
        
        return self.get_data_output(X,A,Y, train_idx, idx)

    def import_employment(self, binarize_conti_sen=False):
        '''
        Senstivie attributes list : AGEP, SEX
        Note RAC1P is dropped 
        Output = [train_ds, (ValiA, ValiY, ValiX), (xdim,adim)]
        '''
        
        sensitive_attrs = self.sensitive_attrs
        
        data =  pd.read_csv(self.data_dir + 'ACSEmployment_CA.csv', sep=",")
        scaler = MinMaxScaler(feature_range=(-1,1))

#         conti_vars = ['bmi','children']
        cate_vars = ['SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC','NATIVITY', 'DEAR', 'DEYE', 'DREM']
        label_var = 'ESR'
            
        # Preprocessing # 
        data = data.assign(SEX = [1  if x==1 else 0 for x in data['SEX']]) # 1:male  - 2:femal
        
#         conti_vars_training = [x for x in conti_vars if x not in sensitive_attrs]
        cate_vars_training = [x for x in cate_vars if x not in sensitive_attrs]
        
#         sensitive_df = pd.get_dummies(data[sensitive_attrs], drop_first=True)
        
        if binarize_conti_sen==False:
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
                
        elif binarize_conti_sen==True:
            if 'AGEP' in sensitive_attrs:
                avg_age = np.mean(data['AGEP'])
                data = data.assign(AGEP = [1 if x > avg_age else 0 for x in data['AGEP']])
                sensitive_df = pd.get_dummies(data[sensitive_attrs], drop_first=True)
                
        
        X = np.array(pd.get_dummies(data[cate_vars_training].astype(str), drop_first=True))
#         X = np.concatenate((XC,XD), axis=1)
        Y = np.array(data[[label_var]])
        A = np.array(sensitive_df)
        
        np.random.seed(self.seed)
        idx = np.random.choice([x for x in range(data.shape[0])], size = int(data.shape[0]*self.vali_ratio), replace=False)
        train_idx = list(set([x for x in range(data.shape[0])]) - set(idx))
        
        return self.get_data_output(X,A,Y, train_idx, idx)
    
    
    def import_crime(self, binarize_conti_sen=False):
        '''
        Sensitive = 'racepctblack'
        '''
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
  
    def import_compas(self, binarize_conti_sen=False):
        """ Downloads COMPAS data from the propublica GitHub repository.
        :return: pandas.DataFrame with columns 'sex', 'age', 'juv_fel_count', 'juv_misd_count',
           'juv_other_count', 'priors_count', 'two_year_recid', 'age_cat_25 - 45',
           'age_cat_Greater than 45', 'age_cat_Less than 25', 'race_African-American',
           'race_Caucasian', 'c_charge_degree_F', 'c_charge_degree_M'
           
           Senstivie attributes list : age, sex, cau
        """
        sensitive_attrs = self.sensitive_attrs # ['sex','cau'] # age, sex, caus self.sensitive_attrs

        data = pd.read_csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")  # noqa: E501
        # filter similar to
        # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        data = data[(data['days_b_screening_arrest'] <= 30) &
                    (data['days_b_screening_arrest'] >= -30) &
                    (data['is_recid'] != -1) &
                    (data['c_charge_degree'] != "O") &
                    (data['score_text'] != "N/A")]
        # filter out all records except the ones with the most common two races
        data = data[(data['race'] == 'African-American') | (data['race'] == 'Caucasian')]
        # Select relevant columns for machine learning.
        # We explicitly leave in age_cat to allow linear classifiers to be non-linear in age
        data = data[["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                     "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"]]
        # map string representation of feature "sex" to 0 for Female and 1 for Male
        data = data.assign(sex=(data["sex"] == "Male") * 1)
        data = pd.get_dummies(data)

        data = data.rename(columns={'race_African-American':'aa','race_Caucasian':'race'})    
        predictor_attrs = [x for x in data.columns if x not in ['sex','age','two_year_recid','aa','cau']]
        
        np.random.seed(self.seed)
        idx = np.random.choice([x for x in range(data.shape[0])], size = int(data.shape[0]*self.vali_ratio), replace=False)
        train_idx = list(set([x for x in range(data.shape[0])]) - set(idx))

        if 'age' in sensitive_attrs:

#             data_sensi_cate = [x for x in ['cau','sex'] if x in sensitive_attrs]
#             sensitive_df = data[['age']]

#             if len(data_sensi_cate) > 0:
            sensitive_df = data[sensitive_attrs]

        else:
            sensitive_df = data[sensitive_attrs]
            # predictor_attrs.append('age')

        A = np.array(sensitive_df)
        X = np.array(data[predictor_attrs])
        Y = np.array(data[['two_year_recid']])

        return self.get_data_output(X,A,Y, train_idx, idx)


    def import_credit(self, binarize_conti_sen=False):
        '''
        Senstivie attributes list : AGE, SEX
        binary output
        Output = [train_ds, (ValiA, ValiY, ValiX), (xdim,adim)]
        '''

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

        
        if binarize_conti_sen==False:
            if 'AGE' in sensitive_attrs:
                data_sensi_cate = [x for x in ['SEX'] if x in sensitive_attrs]
                sensitive_df = pd.DataFrame(scaler.fit_transform(data[['AGE']]))

                if len(data_sensi_cate) > 0:
                    discrete_df = pd.get_dummies(data[data_sensi_cate], drop_first=True)        
                    sensitive_df = pd.concat([sensitive_df, discrete_df],axis=1)
            else:
                sensitive_df = pd.get_dummies(data[sensitive_attrs], drop_first=True)

        elif binarize_conti_sen==True:
            if 'AGE' in sensitive_attrs:
                avg_age = np.mean(data['AGE'])
                data = data.assign(AGE = [1 if x > avg_age else 0 for x in data['AGE']])
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

    def import_law(self, binarize_conti_sen=False):
        '''
        Senstivie attributes list : GPA / Race / Gender
        binary output
        Output = [train_ds, (ValiA, ValiY, ValiX), (xdim,adim)]
        '''

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

