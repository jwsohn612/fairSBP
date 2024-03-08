import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from scipy import stats
from sklearn.metrics import precision_score

def get_average(x):
    if len(x)==0:
        return 0
    else:
        return np.mean(x)
    
def basic_metrics(i,plam, trloss,teloss,cptime,gpu_name):
    return pd.DataFrame({"iter":[i],"plam":[plam],"train_loss":[trloss],"test_loss":[teloss],"cptime":[cptime],"gpu":[gpu_name]})


def positive_predictive_parity(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    return precision

def get_quantile_score(ypred, name):
    quant = np.quantile(ypred, q=[(x+1)/10 for x in range(10)])
    quant_df = pd.DataFrame(quant).T
    quant_df.columns = [name+"_Q{}".format(x+1) for x in range(10)]
    return quant_df

def get_quantile_score2(ypred, name):
    quant = np.quantile(ypred, q=[0.1,0.5,0.9])
    quant_df = pd.DataFrame(quant).T
    quant_df.columns = [name+"_Q{}".format(int(x*100)) for x in [0.1,0.5,0.9]]
    return quant_df

def get_density_matching_scores(df_list, name):
    MaxMin = pd.DataFrame(pd.concat(df_list, axis=0).apply(lambda x:(np.max(x)-np.min(x))/np.std(x), axis=0)).T
    MaxMin.columns = MaxMin.columns.str.replace('B', 'MaxMin_'+name)
    Mean = pd.DataFrame(pd.concat(df_list, axis=0).apply(lambda x:np.mean(np.abs(x - np.mean(x))/np.std(x)), axis=0)).T
    Mean.columns = Mean.columns.str.replace('B', 'Mean_'+name)

    return MaxMin, Mean

def evaluation_metrics(i, VA, VY, ypred, var_loc):
    
    # Threshold
    #cutoff = np.mean(VY)
    VY = tf.reshape(VY, shape = (-1))
    
    ypred = ypred.numpy().reshape(-1)
        
    # Find threshold that maximizes AUC
    fpr, tpr, thresholds = roc_curve(VY, ypred)
    optimal_idx = np.argmax(tpr - fpr) # Youden's J statistic.
    cutoff = thresholds[optimal_idx]
    
    # Marginal & Conditional Scores
    # DP
    ypredA1 = ypred[VA[:,var_loc]==1]
    ypredA0 = ypred[VA[:,var_loc]==0]
    aY = np.mean(ypred)
    aYA1 = np.mean(ypredA1)
    aYA0 = np.mean(ypredA0)
    aY1 = np.mean(ypred[VY==1])
    aY0 = np.mean(ypred[VY==0])
    
    KSA1,  KSA1pval = stats.ks_2samp(ypredA1, ypred)
    KSA0,  KSA0pval = stats.ks_2samp(ypredA0, ypred)
    
    # QuantScore
    aYQ = get_quantile_score(ypred, "aY")
    aYA1Q = get_quantile_score(ypredA1, "aYA1")
    aYA0Q = get_quantile_score(ypredA0, "aYA0")
    
    # EO
    ypredYA1Y1 = ypred[(VA[:,var_loc]==1)&(VY==1)]
    ypredYA0Y1 = ypred[(VA[:,var_loc]==0)&(VY==1)]
    ypredYA1Y0 = ypred[(VA[:,var_loc]==1)&(VY==0)]
    ypredYA0Y0 = ypred[(VA[:,var_loc]==0)&(VY==0)]
    
    aYA1Y1 = np.mean(ypredYA1Y1)
    aYA0Y1 = np.mean(ypredYA0Y1)
    aYA1Y0 = np.mean(ypredYA1Y0)
    aYA0Y0 = np.mean(ypredYA0Y0)
    
    aYA1Y1Q = get_quantile_score(ypredYA1Y1, "aYA1Y1")
    aYA0Y1Q = get_quantile_score(ypredYA0Y1, "aYA0Y1")
    aYA1Y0Q = get_quantile_score(ypredYA1Y0, "aYA1Y0")
    aYA0Y0Q = get_quantile_score(ypredYA0Y0, "aYA0Y0")
    
    KSY1A1,  KSY1A1pval = stats.ks_2samp(ypredYA1Y1, ypred[VY==1])
    KSY1A0,  KSY1A0pval = stats.ks_2samp(ypredYA0Y1, ypred[VY==1])
    KSY0A1,  KSY0A1pval = stats.ks_2samp(ypredYA1Y0, ypred[VY==0])
    KSY0A0,  KSY0A0pval = stats.ks_2samp(ypredYA0Y0, ypred[VY==0])
    
    info3 = pd.DataFrame({'KSA1':[KSA1],'KSA1pval':[KSA1pval],
                          'KSA0':[KSA0],'KSA0pval':[KSA0pval],
                         'KSY1A1':[KSY1A1],'KSY1A1pval':[KSY1A1pval],'KSY1A0':[KSY1A0],'KSY1A0pval':[KSY1A0pval],
                         'KSY0A1':[KSY0A1],'KSY0A1pval':[KSY0A1pval],'KSY0A0':[KSY0A0],'KSY0A0pval':[KSY0A0pval],
                         })
    
    info2 = pd.concat([aYQ,aYA1Q, aYA0Q, aYA1Y1Q, aYA0Y1Q, aYA1Y0Q, aYA0Y0Q],axis=1)
    
    # BER 
    ylabel = (ypred>cutoff).astype(float)
    ytrue = VY.numpy().reshape(-1)
    
    idx1 = VA[:,var_loc].numpy()==1
    idx0 = VA[:,var_loc].numpy()==0
    
    ytrueA1 =  ytrue[idx1]
    ypredA1 = ypred[idx1]
    ylabelA1 = ylabel[idx1]
    ytrueA0 =  ytrue[idx0]
    ypredA0 = ypred[idx0]
    ylabelA0 = ylabel[idx0]
    
    # Sufficiency 
    YA1 = positive_predictive_parity(ytrueA1, ylabelA1)
    YA0 = positive_predictive_parity(ytrueA0, ylabelA0)
    
    # Conditional Probabilities
    # DP
    pY = np.mean(ylabel)
    pYA0 = np.mean(ylabel[VA[:,var_loc]==0])
    pYA1 = np.mean(ylabel[VA[:,var_loc]==1])
        
    # EO 
    pYA0Y1 = np.mean(ylabel[(VA[:,var_loc]==0)&(VY==1)])
    pYA1Y1 = np.mean(ylabel[(VA[:,var_loc]==1)&(VY==1)])
    pYA0Y0 = np.mean(ylabel[(VA[:,var_loc]==0)&(VY==0)])
    pYA1Y0 = np.mean(ylabel[(VA[:,var_loc]==1)&(VY==0)])
    
    pY1 = np.mean(ylabel[VY==1])
    pY0 = np.mean(ylabel[VY==0])
    
    # Sufficiency?
    # sufscore = suffiency_metric(ypred, VA, var_loc)
    f1 = f1_score(ytrue, ylabel)
    
    # AUC
    auc = roc_auc_score(ytrue, ypred)
    
    info1 = pd.DataFrame({'iter':[i], 'f1':[f1],'auc':[auc], # 'suf':[subscore], 
                          'YA1':[YA1],'YA0':[YA0], 
                         'aY':[aY], 'aYA1':[aYA1], 'aYA0':[aYA0], 'aY1':[aY1], 'aY0':[aY0], 
                         'aYA0Y1':[aYA0Y1],'aYA1Y0':[aYA1Y0],'aYA1Y1':[aYA1Y1],'aYA0Y0':[aYA0Y0],
                         'pY':[pY], 'pYA1':[pYA1], 'pYA0':[pYA0], 'pY1':[pY1], 'pY0':[pY0],
                         'pYA0Y1':[pYA0Y1],'pYA1Y0':[pYA1Y0],'pYA1Y1':[pYA1Y1],'pYA0Y0':[pYA0Y0]})
    
    return pd.concat([info1,info3,info2],axis=1)

def get_quanbased_ks_score(ypred, VA, va_quants, varloc, name, counts=10):
    
    output_stat = [stats.ks_2samp(ypred[VA[:,varloc]<=x], ypred)[0] for x in va_quants] 
    output_pval = [stats.ks_2samp(ypred[VA[:,varloc]<=x], ypred)[1] for x in va_quants] 
    output_stat = pd.DataFrame(output_stat).T
    output_pval = pd.DataFrame(output_pval).T
    
    output_stat.columns = [name+"_S{}".format(x+1) for x in range(counts)]
    output_pval.columns = [name+"_P{}".format(x+1) for x in range(counts)]
    return pd.concat([output_stat, output_pval],axis=1)

def evaluation_metrics_cont(i, VA, VY, ypred, var_loc):
    '''
    continuous A 
    discrete Y
    '''
    VY = tf.reshape(VY, shape = (-1))
    
    ypred = ypred.numpy().reshape(-1)
        
    # Find threshold that maximizes AUC
    fpr, tpr, thresholds = roc_curve(VY, ypred)
    optimal_idx = np.argmax(tpr - fpr) # Youden's J statistic.
    cutoff = thresholds[optimal_idx]
    
    n_quants = 5
    
    quant = np.quantile(VA[:,var_loc].numpy(), q=[(x+1)/n_quants for x in range(n_quants)])
    
    # Independence
    KSYAa = get_quanbased_ks_score(ypred=ypred, VA=VA, va_quants=quant, varloc=var_loc, name = "KSYAa", counts=n_quants)    
    
    # Separation
    KSY1Aa = get_quanbased_ks_score(ypred=ypred[VY==1], VA=VA[VY==1], va_quants=quant, varloc=var_loc, name="KSY1Aa", counts=n_quants)
    KSY0Aa = get_quanbased_ks_score(ypred=ypred[VY==0], VA=VA[VY==0], va_quants=quant, varloc=var_loc, name="KSY0Aa", counts=n_quants)
    
    KS = pd.concat([KSYAa, KSY1Aa, KSY0Aa],axis=1)
    
    quant2 = np.quantile(VA[:,var_loc].numpy(), q=[(x+1)/n_quants for x in range(n_quants)])
    
    # Independence
    aYAa = pd.concat([get_quantile_score2(ypred[VA[:,var_loc] <= x],name = "aYAa") for x in quant2] ,axis=1)
    # Separation
    aY0Aa = pd.concat([get_quantile_score2(ypred[(VA[:,var_loc] <= x)&(VY==0)],name="aY0Aa") for x in quant2] ,axis=1)
    aY1Aa = pd.concat([get_quantile_score2(ypred[(VA[:,var_loc] <= x)&(VY==1)],name="aY1Aa") for x in quant2] ,axis=1)
    
    # sufscore = suffiency_metric(ypred, VA, var_loc)
    ylabel = (ypred>cutoff).astype(float)
    ytrue = VY.numpy().reshape(-1)
    f1 = f1_score(ytrue, ylabel)
    auc = roc_auc_score(ytrue, ypred)
    
    # Independence
    pYAa = np.mean([np.abs(np.mean(ylabel[VA[:,var_loc] <= x]) / np.mean(ylabel) -1) for x in quant2])
    
    # Separation
    pY0Aa = np.mean([np.abs(np.mean(ylabel[(VA[:,var_loc] <= x)&(VY==0)])/np.mean(ylabel[VY==0])-1) for x in quant2])
    pY1Aa = np.mean([np.abs(np.mean(ylabel[(VA[:,var_loc] <= x)&(VY==1)])/np.mean(ylabel[VY==1])-1) for x in quant2])

    SP = pd.concat([aYAa, aY1Aa, aY0Aa],axis=1)
    

    info = pd.DataFrame({'iter':[i], 'f1':[f1],'auc':[auc], 'pYAa':[pYAa], "pY0Aa":[pY0Aa], "pY1Aa":[pY1Aa]})
    
    return  pd.concat([info,KS,SP],axis=1)



def get_ks_statistis(VY, VA, ypred, y, a):
    '''
    Assume y and a are given 
    '''
    target = ypred[(VY<=y)&(VA<=a)]
    comparison = ypred[(VY<=y)]
    
    if len(target)==0:
        return None 
    else:
        return stats.ks_2samp(target, comparison)[0]
    
def get_sum_statistis(VY, VA, ypred, y, a):
    '''
    Assume y and a are given 
    '''
    target = ypred[(VY<=y)&(VA<=a)]
    comparison = ypred[(VY<=y)]
    
    if len(target)==0:
        return None 
    else:
        return np.abs(np.mean(ypred[(VY<=y)&(VA<=a)])/np.mean(ypred[(VY<=y)])-1)
    
def make_df_ksstat_conti(VY, VA, ypred, quants_Y, quants_A, ranges):
    output = pd.DataFrame([np.mean([get_ks_statistis(VY, VA, ypred, y, a) for a in quants_A]) for y in quants_Y]).T
    output.columns = ["KSYyAa{}".format(int(y*100)) for y in ranges]
    return output

def make_df_sumstat_conti(VY, VA, ypred, quants_Y, quants_A, ranges):
    output = pd.DataFrame([np.mean([get_sum_statistis(VY, VA, ypred, y, a) for a in quants_A]) for y in quants_Y]).T
    output.columns = ["aYyAa{}".format(int(y*100)) for y in ranges]
    return output

def evaluation_metrics_contout(i, VA, VY, ypred, var_loc, util):
    '''
    continuous A 
    continuous Y
    '''
    ranges = [(x+1)/9 for x in range(9)]
    counts = len(ranges)
    
    VY = tf.reshape(VY, shape = (-1))
    quants_Y = np.quantile(VY, q=ranges)
    quants_A = np.quantile(VA.numpy().reshape(-1), q=ranges)
    
    ypred = ypred.numpy().reshape(-1)
    
    quant = np.quantile(VA[:,var_loc].numpy(), q=ranges)
    
    # Independence
    KSYAa = get_quanbased_ks_score(ypred=ypred, VA=VA, va_quants=quants_A, varloc=var_loc, name = "KSYAa", counts=counts)    
    
    # Separation
    KSYyAa = make_df_ksstat_conti(VY.numpy(), VA.numpy().reshape(-1), ypred, quants_Y, quants_A, ranges)
    
    KS = pd.concat([KSYAa, KSYyAa],axis=1)
    
    quant2 = np.quantile(VA[:,var_loc].numpy(), q=ranges)
    
    # Independence
    aYAa = pd.DataFrame({"aYAa":[np.mean([np.abs(np.mean(ypred[VA[:,var_loc] <= x])/np.mean(ypred)-1) for x in quants_A])]})

    # Separation
    aYyAa = make_df_sumstat_conti(VY.numpy(), VA.numpy().reshape(-1), ypred,quants_Y,quants_A, ranges)   
    
    SumStat = pd.concat([aYAa, aYyAa],axis=1)
    output = pd.concat([KS, SumStat],axis=1)
    output = output.assign(iter= i, testloss = util)
    return output

def evaluation_metrics_contout_disca(i, VA, VY, ypred, var_loc, util):
    '''
    discrete A 
    continuous Y
    '''
    ranges = [(x+1)/9 for x in range(9)]
    counts = len(ranges)
    
    VY = tf.reshape(VY, shape = (-1))
    quants_Y = np.quantile(VY, q=ranges)
        
    ypred = ypred.numpy().reshape(-1)
    
    # A==0
    # A==1
    target_A = VA[:,var_loc]
    ypredYA1 = ypred[target_A==1]
    ypredYA0 = ypred[target_A==0]
    
    # SP
    YA1 = np.mean(ypredYA1)
    YA0 = np.mean(ypredYA0)
    
    KSY1A1,  KSY1A1pval = stats.ks_2samp(YA1, ypred[VY==1])
    KSY1A0,  KSY1A0pval = stats.ks_2samp(YA0, ypred[VY==1])
    KSY0A1,  KSY0A1pval = stats.ks_2samp(ypredYA1Y0, ypred[VY==0])
    KSY0A0,  KSY0A0pval = stats.ks_2samp(ypredYA0Y0, ypred[VY==0])
    
    # Independence
    KSYAa = get_quanbased_ks_score(ypred=ypred, VA=VA, va_quants=quants_A, varloc=var_loc, name = "KSYAa", counts=counts)    
    
    # Separation
    KSYyAa = make_df_ksstat_conti(VY.numpy(), VA.numpy().reshape(-1), ypred, quants_Y, quants_A, ranges)
    
    KS = pd.concat([KSYAa, KSYyAa],axis=1)
    
    quant2 = np.quantile(VA[:,var_loc].numpy(), q=ranges)
    
    # Independence
    aYAa = pd.DataFrame({"aYAa":[np.mean([np.abs(np.mean(ypred[VA[:,var_loc] <= x])/np.mean(ypred)-1) for x in quants_A])]})

    # Separation
    aYyAa = make_df_sumstat_conti(VY.numpy(), VA.numpy().reshape(-1), ypred,quants_Y,quants_A, ranges)   
    
    SumStat = pd.concat([aYAa, aYyAa],axis=1)
    output = pd.concat([KS, SumStat],axis=1)
    output = output.assign(iter= i, testloss = util)
    return output