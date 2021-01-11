import numpy as np
import pandas as pd
import os.path
import logging
import pyarrow
import pickle as p
import gc
import re
import seaborn
import matplotlib
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_rows', 1000)

import sys
from time import time, strftime, localtime
import pandas.core.algorithms as algos
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances

from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# np.seterr(divide='ignore', invalid='ignore')
import scenario



def secondsToStr(elapsed=None):
    '''
    Returns 
    '''
    if elapsed is None:
        return strftime("%Y-%m-%d_%H_%M_%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def startLogger():
    log_name = "../outputs/logs/" + "variable_reduction_" + secondsToStr() + ".log"
    logging.basicConfig(filename=log_name, filemode="w",
                        format="%(asctime)s - %(message)s", level=logging.INFO)

def CreateVarTypeSplit(df, cols=[]):
    """
    Simplistic function that splits df variables to categorical and continuous 
    simply based on # of unique values: if have only two values, then group as categorical
    Parameters:
                    df - dataframe
                  cols - a list of columns we want to split from
                         it can also be specify through the scenario file by "custom_variableReduction_ls"
    Returns:
           cate_var_ls - a list of categorical vars
           cont_var_ls - a list of continuous vars
    Writes:
           cate_var_ls - a list of categorical vars
           cont_var_ls - a list of continuous vars
    """
    logging.info("-----------------------------------------------------------------------------------")
    logging.info("Split Vars to Cate and Cont")
    logging.info("-----------------------------------------------------------------------------------")
    if scenario.custom_variableReduction_ls:
        logging.info(
            'Using custom input as the list of variables to splits from')
        cols = scenario.custom_variableReduction_ls
    else:
        logging.info(
            'Using all input df columns as the list of variables to splits from')

    cols = [x for x in cols if x != scenario.target]
    logging.info("Make sure we exclude the target variable: {0}".format(
        scenario.target not in cols))
    logging.info("Starting splitting Vars to Cate and Cont")
    cate_var_ls = df[cols].nunique()
    cate_var_ls = list(set(cate_var_ls.loc[cate_var_ls == 2].index))
    for var in cate_var_ls:
        if set(df[var].value_counts().index) != {0, 1}:
            cate_var_ls.remove(var)
    cont_var_ls = [x for x in cols if x not in cate_var_ls]

    try: 
        cont_var_ls.remove(scenario.sample_split_variable)
    except ValueError:
        pass
    try: 
        cate_var_ls.remove(scenario.sample_split_variable)
    except ValueError:
        pass

    logging.info(
        "Categorical vars for variable reduction written to /outputs/most_recent/variable_reduction/cate_var_ls.csv")
    pd.DataFrame(cate_var_ls).to_csv(
        "./outputs/most_recent/variable_reduction/cate_var_ls.csv", index=False)
    logging.info(
        "Continuous vars for variable reduction written to /outputs/most_recent/variable_reduction/cont_var_ls.csv")
    pd.DataFrame(cont_var_ls).to_csv(
        "./outputs/most_recent/variable_reduction/cont_var_ls.csv", index=False)

    logging.info("There are {0} categorical variables (0/1 type). For column names, see the 'cate_var_ls'".format(
        len(cate_var_ls), cate_var_ls))
    logging.info("There are {0} continuous variables. For column names, see the 'cont_var_ls'".format(
        len(cont_var_ls), cont_var_ls))
    logging.info("Finished splitting Vars to Cate and Cont")

    return cate_var_ls, cont_var_ls

def correlation_reduction_worker(corrMatrix, target, threshold, reductionMetrics=''):
    '''
    This function works to drop vars over threshold and heat map a pre and post correlation matrix of any sorts.
    The threshold can also be specify through the scenario file by "custom_correlationReduction_thresh"
    Parameters:
               corrMatrix - correlation matrix
                   target - dataframe containing variables' correlation with your target variable
                threshold - your threshold for variable reduction
         reductionMetrics - one of four correlation reduction metrics:
                            Pearson, Spearmans, EuclideanDistances or LogisticRegression
    Returns:
              droppedInfo - dataFrame of variables that were dicarded
                  corrmat - post correlation matrix of any sorts
    Writes:
              droppedInfo - dataFrame of variables that were dicarded
                  corrmat - post correlation matrix of any sorts
    '''

    drop = []
    seaborn.set(context="paper", font="monospace")
    corrmat = abs(corrMatrix.copy(deep=True))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 8))
    ax.set_yticks([])
    # Draw the heatmap using seaborn
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    with seaborn.axes_style("white"):
        seaborn.heatmap(corrmat, vmax=1, mask=mask, square=True,
                        xticklabels=20, yticklabels=20, cmap='Blues')
        ax.set_title("{0} - Original Correlation Matrix".format(reductionMetrics))

    if not os.path.isfile('../outputs/variable_reduction/{}_Original.png'.format(reductionMetrics)):
        f.savefig('../outputs/variable_reduction/{}_Original.png'.format(reductionMetrics))
    

    # Correlation Threshold
    mid = target.drop(target.index[0], 1).T
    mid['target_abs'] = abs(mid[target.index[0]])
    mid.sort_values('target_abs', inplace=True, ascending=False)
    order = list(mid.index)
    corrMatrix = corrMatrix.reindex(order, axis=1)
    corrMatrix = corrMatrix.reindex(order)
    column_list = corrMatrix.keys()###
    for col in column_list:###
        # This is a temporary drop list that'll be emptied in each iteration, we use it to store the dropped vars and apply to corrMatrix drop
        drop_tmp = [] 
        if col in corrMatrix.keys(): # corrMatrix does inplace drop of columns and rows in each iteration, need to check whether col is still in corrMatrix
            for i in range(len(corrMatrix)): # Iterating through all the vars
                if col != corrMatrix.keys()[i]: # Make sure col is not compared with itself
                    if abs(corrMatrix[col][i]) > threshold: # Correlation Threshold Checking
                        # This is a cumulative drop list that'll be used as the output to show the summary of drop vars
                        # Append [dropped_var, col, corr(dropped_var, target), corr(col, target)]
                        drop.append([corrMatrix.keys()[i],col,target[corrMatrix.keys()[i]].values[0], target[col].values[0]])
                        drop_tmp.append(corrMatrix.keys()[i]) # Append dropped_var
            corrMatrix.drop(drop_tmp,axis=1,inplace=True) # Drop columns
            corrMatrix.drop(drop_tmp, axis=0, inplace=True) # Drop rows
    corrmat = abs(corrMatrix.copy(deep=True))

    droppedInfo = pd.DataFrame(drop, columns=['discarded_variable', 'correlated_to', 'discarded_variable_correlation_to_target', 'correlated_to_correlation_to_target'])

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 8))
    ax.set_yticks([])
    # Draw the heatmap using seaborn
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    with seaborn.axes_style("white"):
        seaborn.heatmap(corrmat, vmax=1, mask=mask, square=True,
                        xticklabels=20, yticklabels=20, cmap='Blues')
        ax.set_title("{} - Post Reduction Correlation Matrix - {}".format(reductionMetrics,threshold))
    
    f.savefig('../outputs/variable_reduction/{}_Post_Reduction_Correlation_Matrix_{}.png'.format(reductionMetrics, str(threshold).replace('.','')))
    
    return corrmat, droppedInfo


def CreatePearsonReduction(df, vars_ls=[], target_var=[], thresh=0.75):
    '''
    This function calculates pearson coeffiencient matrix that is passed to the "correlation_reduction_worker" function. 
    The threshold can also be specify through the scenario file by "custom_pearsonCorrelation_thresh"
    Parameters: 
                      df - dataframe you want to start pearson correlation reduction from
                 vars_ls - list of your continuous variables
              target_var - list of your target variable
               threshold - your threshold for pearson correlation reduction
    Returns:
                      df - dataframe after pearson correlation reduction 
          reducedCorrmat - post pearson correlation reduction matrix of continuous vars
                 dropped - dataFrame of variables that were dicarded
    Writes:
                npMatrix - original pearson correlation matrix of the input continuous vars
                 dropped - dataFrame of variables that were dicarded
    '''

    logging.info("-----------------------------------------------------------------------------------")
    logging.info("Correlation Reduction for Train: Pearson")
    logging.info("-----------------------------------------------------------------------------------")
    logging.info("Starting Pearson correlation reduction for {0} vars".format(len(vars_ls)))
    if target_var[0] not in vars_ls: 
        ls = vars_ls + target_var
    else:
        ls = vars_ls
    warnings.filterwarnings("error")


    if not os.path.isfile('../outputs/variable_reduction/Pearson_originalCorrmat.csv'):    
        try:
            npMatrix = pd.DataFrame(np.corrcoef(df[ls], rowvar=0))
        except RuntimeWarning as e:
            print("Check that train_processed.parquet.gzip does not contain unary values")
            raise
        npMatrix.columns = ls
        npMatrix.index = ls
        logging.info("Original pearson correlation written to ../outputs/variable_reduction/Pearson_originalCorrmat.csv")
        npMatrix.to_csv("../outputs/variable_reduction/Pearson_originalCorrmat.csv")
    else:
        npMatrix = pd.read_csv("../outputs/variable_reduction/Pearson_originalCorrmat.csv")
        npMatrix = npMatrix.set_index('Unnamed: 0')
        npMatrix.index.name = None
    
    
    corrMatrix = npMatrix.copy(deep=True)
    target = corrMatrix[target_var].T
    corrMatrix.drop(target_var, axis=1, inplace=True)
    corrMatrix.drop(target_var, axis=0, inplace=True)
    logging.info('Start with: %s variables' % corrMatrix.shape[0])

    # now use the worker function
    logging.info("Starting using the worker function to start variable reduction")
    reducedCorrmat, dropped = correlation_reduction_worker(corrMatrix, target, thresh, "Pearson")
    logging.info("Finished using the worker function")

    # Drop variables from input df
    logging.info("Dropped variables")
    df = df.drop(dropped.discarded_variable.tolist(), axis=1)

    # save output
    logging.info("Reduced pearson correlation written to ../outputs/variable_reduction/Pearson_reducedCorrmat_{}.csv".format(str(thresh).replace('.','')))
    reducedCorrmat.to_csv("../outputs/variable_reduction/Pearson_reducedCorrmat_{}.csv".format(str(thresh).replace('.','')))

    logging.info("Dropped variables by pearson correlation written to ../outputs/variable_reduction/Pearson_dropped_ls_{}.csv".format(str(thresh).replace('.','')))
    dropped.to_csv("../outputs/variable_reduction/Pearson_dropped_ls_{}.csv".format(str(thresh).replace('.','')),index = False)

    logging.info("Using threshold {0}, dropped {1} cont variables. For column names, see the 'dropped' list: {2}".format(thresh, len(dropped.discarded_variable.tolist()), dropped.discarded_variable.tolist()))
    logging.info('Left {0} variables'.format(reducedCorrmat.shape[0]))
    logging.info("Finished pearson correlation reduction")
    
    return df, dropped

def ApplyPearsonReduction(df, pearson_droppedVars=[]):
    '''
    This function applys the same Pearson Reduction as your train for your test/validation
    Parameters: 
                      df - dataframe: your test/validation
     pearson_droppedVars - list of dropped columns from train data from CreatePearsonReduction function
    Returns:
                      df - dataframe after pearson correlation reduction 
    '''
    logging.info(
        "-----------------------------------------------------------------------------------")
    logging.info("Correlation Reduction for NonTrain: Cont-Cont (Pearson)")
    logging.info(
        "-----------------------------------------------------------------------------------")
    logging.info("Starting pearson correlation reduction")

    # drop vars
    df = df.drop(pearson_droppedVars, axis=1)
    logging.info("Dropped {0} columns based on pearson correlation. For column names, see the 'pearson_droppedVars' list: {1}".format(
        len(pearson_droppedVars), pearson_droppedVars))
    logging.info(
        "Finished pearson correlation reduction")
    return df


def CreateSpearmansReduction(df, vars_ls=[], target_var=[], thresh=0.75, pval_thresh=0.05):
    '''
    This function calculate spearmans coeffiencient matrix that is passed to the "correlation_reduction_worker" function. 
    The threshold can also be specify through the scenario file by "custom_spearmansCorrelation_thresh"
    Parameters: 
                      df - dataframe you want to start spearmans correlation reduction from
                 vars_ls - list of your continuous variables
              target_var - list of your target variable
               threshold - your threshold for spearmans correlation reduction
    Returns:
                      df - dataframe after spearmans correlation reduction 
          reducedCorrmat - post spearmans correlation reduction matrix of continuous vars
                 dropped - dataFrame of variables that were dicarded
    Writes:
                npMatrix - original spearmans correlation matrix of the input continuous vars
                 dropped - dataFrame of variables that were dicarded
    '''
    
    logging.info("-----------------------------------------------------------------------------------")
    logging.info("Correlation Reduction for Train: Spearmans")
    logging.info("-----------------------------------------------------------------------------------")
    logging.info("Starting spearmans correlation reduction for {0} vars".format(len(vars_ls)))
    if target_var[0] not in vars_ls: 
        ls = vars_ls + target_var
    else:
        ls = vars_ls
    

    if not os.path.isfile('../outputs/variable_reduction/Spearmans_originalCorrmat.csv'):    
        npMatrix, pvalMatrix = spearmanr(df[ls], axis=0)
        # fail to reject null hypothesis, uncorrelated, default pval = 0.05, i.e.95% conf
        npMatrix[pvalMatrix > pval_thresh] = 0
        npMatrix = pd.DataFrame(npMatrix)
        npMatrix.columns = ls
        npMatrix.index = ls
        logging.info("Original spearmans correlation written to ../outputs/variable_reduction/Spearmans_originalCorrmat.csv")
        npMatrix.to_csv("../outputs/variable_reduction/Spearmans_originalCorrmat.csv")
    else:
        npMatrix = pd.read_csv("../outputs/variable_reduction/Spearmans_originalCorrmat.csv")
        npMatrix = npMatrix.set_index('Unnamed: 0')
        npMatrix.index.name = None


    corrMatrix = npMatrix.copy(deep=True)
    target = corrMatrix[target_var].T

    corrMatrix.drop(target_var, axis=1, inplace=True)
    corrMatrix.drop(target_var, axis=0, inplace=True)

    logging.info("Starting creating target corr with each variable using logistic regression")
    # create target corr with each variable using logistic regression
    # since target is categorical and variables are continious.
    if not os.path.isfile('../outputs/variable_reduction/Spearmans_target_auc.p'):
        for i in ls:
            model = LogisticRegression(class_weight='balanced', penalty='l2', C=0.001,max_iter=4000)
            model.fit(StandardScaler().fit_transform(df[i].values.reshape(-1, 1)),
                    df[target_var].values.ravel())
            target[i] = roc_auc_score(y_score=model.predict_proba(df[i].values.reshape(-1, 1))[:, 1],
                                    y_true=df[target_var].values.ravel())
        with open('../outputs/variable_reduction/Spearmans_target_auc.p', 'wb') as f:
            p.dump(target, f)
        logging.info("Finished creating target corr with each variable using logistic regression")
    else:
        with open('../outputs/variable_reduction/Spearmans_target_auc.p', 'rb') as f:
            target = p.load(f)

    # now use the worker function to start variable reduction
    logging.info("Starting using the worker function to start variable reduction")
    reducedCorrmat, dropped = correlation_reduction_worker(corrMatrix, target, thresh, "Spearmans")
    logging.info("Finished using the worker function")

    # Drop varialbes from input df
    logging.info("Dropped variables")
    df = df.drop(dropped.discarded_variable.tolist(), axis=1)

    # save output
    logging.info("Reduced spearman correlation written to ../outputs/variable_reduction/Spearmans_reducedCorrmat_{}.csv".format(str(thresh).replace('.','')))
    reducedCorrmat.to_csv("../outputs/variable_reduction/Spearmans_reducedCorrmat_{}.csv".format(str(thresh).replace('.','')))

    logging.info("Dropped variables by spearmans correlation written to ../outputs/variable_reduction/Spearmans_dropped_ls_{}.csv".format(str(thresh).replace('.','')))
    dropped.to_csv('../outputs/variable_reduction/Spearmans_dropped_ls_{}.csv'.format(str(thresh).replace('.','')),index=False)

    logging.info("Using threshold {0}, dropped {1} cont variables. For column names, see the 'dropped' list: {2}".format(thresh, len(dropped.discarded_variable.tolist()), dropped.discarded_variable.tolist()))
    logging.info('Left {0} variables'.format(reducedCorrmat.shape[0]))
    logging.info("Finished spearmans correlation reduction")
    
    return df, dropped

def ApplySpearmansReduction(df, spearmans_droppedVars=[]):
    '''
    This function applys the same Spearmans Reduction as your train for your test/validation
    Parameters: 
                      df - dataframe: your test/validation
   spearmans_droppedVars - list of dropped columns from train data from CreateSpearmansReduction function
    Returns:
                      df - dataframe after spearmans correlation reduction 
    '''
    logging.info(
        "-----------------------------------------------------------------------------------")
    logging.info("Correlation Reduction for NonTrain: Cont-Cont (Spearmans)")
    logging.info(
        "-----------------------------------------------------------------------------------")
    logging.info("Starting spearmans correlation reduction")

    # drop vars
    df = df.drop(spearmans_droppedVars, axis=1)
    logging.info("Dropped {0} columns based on spearmans correlation. For column names, see the 'spearmans_droppedVars' list: {1}".format(
        len(spearmans_droppedVars), spearmans_droppedVars))
    logging.info(
        "Finished spearmans correlation reduction")
    return df


if __name__ == '__main__':
    
    startLogger()
    X_train = pd.read_parquet('../outputs/train_df.parquet.gzip')

    for thresh in [0.75, 0.7, 0.65]:
        train_df_reduced, pearson_droppedVars = CreatePearsonReduction(df=X_train,
                                                            vars_ls=list(X_train),
                                                            target_var=['TARGET'],
                                                            thresh=thresh)

        train_df_reduced, spearmans_droppedVars = CreateSpearmansReduction(df=X_train,
                                                        var_ls=list(X_train),
                                                        target_var=['TARGET'],
                                                        thresh=thresh,
                                                        pval_thresh=thresh)


