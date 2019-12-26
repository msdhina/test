# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:55:04 2019

@author: Baymax
"""

import pandas as pd
import time
import numpy as np
import operator
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from scipy.stats import shapiro
from scipy.stats import skew
from scipy.stats import kurtosis
from numpy.random import seed
from numpy.random import randn
from scipy.stats import anderson
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs



def init_initial_analysis(df):
    """
    Given a dataframe produces a simple report on initial data analytics
    Params:
        - df 
    Returns:
        - Shape of dataframe records and columns
        - Columns and data types
    """
    print('Report of Initial Data Analysis:\n')
    print(f'Shape of dataframe: {df.shape}\n')
    print(f'Features and Data Types: \n {df.dtypes}\n')
    print(f' {[df[i].value_counts() for i in df.columns]}\n')
    print(f'Information: {df.info()}\n')
    print(f'Describe: \n {df.describe()}\n')

    
def init_percent_missing(df):
    """
    Given a dataframe it calculates the percentage of missing records per column
    Params:
        - df
    Returns:
        - Dictionary of column name and percentage of missing records
    """
    col=list(df.columns)
    perc=[round(df[c].isna().mean()*100,2) for c in col]
    miss_dict=dict(zip(col,perc))
    return miss_dict
    


def init_numerical_features(df):
    """
    Given the dataframe, select the features that are numerical
    Params:
        - df
    Returns:
        - List of numerical features
    """
    numeric_list=[]
    for c in list(df.columns):
        if (df[c].dtypes) in['int64','int','int32','float','float32','float64']:
            numeric_list.append(c)
    return numeric_list
    

def init_categorical_features(df):
    """
    Given the dataframe, select the features that are categorical
    Params:
        - df
    Returns:
        - List of categorical features
    """
    cat_list=[]
    for c in list(df.columns):
        if (df[c].dtypes) in['object','str']:
            cat_list.append(c)
    return cat_list

def test_normality_test_shapiro(df,numeric_list):
    """
    Given a dataframe determines whether each numerical column is Gaussian 
    Ho = Assumes distribution is Gaussian
    Ha = Assumes distribution is not Gaussian
    p <= alpha: reject H0, not normal.
    p > alpha: fail to reject H0, normal.
    Params:
        - df,num_list
    Returns:
        - W Statistic  Shapiro-Wilk test for normality
        - p-value
        - List of columns that do not have gaussian distribution
    """
    non_gauss=[]
    w_stat=[]
    # Determine if each sample of numerical feature is gaussian
    alpha = 0.05
    for n in numeric_list:
        stat,p=shapiro(df[n])
        sns.distplot(df[n])
        plt.show()
        #print(tuple(skew(df[n]),kurtosis(df[n])))
        print(p)
        if p > alpha:
        	print('Sample looks Gaussian (fail to reject H0)')
        else:
        	print('Sample does not look Gaussian (reject H0)')
        	non_gauss.append(n)
        	w_stat.append(stat)
    # Dictionary of numerical features not gaussian and W-Statistic        
    norm_dict=dict(zip(non_gauss,w_stat))
    print("Shapiro-wilk test :\n", norm_dict)


def test_normality_test_anderson(df,numeric_list):
    seed(1)
    print('\nHo = Assumes distribution is not Gaussian\nHa = Assumes distribution is Gaussian\n\n')
    for n in numeric_list:
        # generate univariate observations
        data = df[n]
        # normality test
        result = anderson(data)
        print(f'AndersonTest for {n}:')
        print('Statistic: %.3f' % result.statistic)
        for i in range(len(result.critical_values)):
        	sl, cv = result.significance_level[i], result.critical_values[i]
        	if result.statistic < result.critical_values[i]:
        		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        	else:
        		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        print('\n\n')
        
def init_correlation(df,target):
    
    a=list(df.columns)
    a.remove(target)
    
    individual_features_df = []
    for i in a: 
        tmpDf = df[[i,target]]
        #tmpDf = tmpDf[tmpDf[df[i]] != 0]
        individual_features_df.append(tmpDf)
    
    all_correlations = {feature.columns[0]: feature.corr()[target][0] for feature in individual_features_df}
    all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
    for (key, value) in all_correlations:
        print("{:>15}: {:>15}".format(key, value))
        
    golden_features_list = [key for key, value in all_correlations if abs(value) >= 0.5 or abs(value)<=-0.5]
    print("\n\nThere is {} strongly correlated values with {}:\n\n{}".format(len(golden_features_list),target,golden_features_list))



def remove_correlation(df,pos_thres,neg_thres):
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] >= pos_thres) or any(upper[column] <= neg_thres)]
    print("Dropping columns:\n",to_drop)    
    # Drop features 
    df.drop(df[to_drop], axis=1)
    

def plot_correlation(df):
    #Using Pearson Correlation
    plt.figure(figsize=(12,10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    

def init_pairplot(df,target):
    for i in range(0, len(df.columns), 5):
        sns.pairplot(data=df,
                    x_vars=df.columns[i:i+5],
                    y_vars=[target])
    





