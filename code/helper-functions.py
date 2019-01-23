# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:39:06 2019

@author: vande70
"""

import os
import csv

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


def cramers_v(cat1='', cat2='', data=''):
    confusion_matrix = pd.crosstab(data[cat1],data[cat2])
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def correlation_ratio(cat, cont, data):
    fcat, _ = pd.factorize(data[cat])
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = data[cont][np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(data[cont],y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta


# Create function to explore some bivariate relationships
def univariate(df_set, x_var):
    if (df_set[x_var].dtype in ["int64", "float64"] and df_set[x_var].nunique() > 10):
            sns.distplot(df_set[x_var])
    elif (df_set[x_var].nunique() <= 10):
            sns.countplot(df_set[x_var])
    else:
        "dtype not recognized or too many categories"
        
def bivariate(df_set, y_var, x_var):
    if (df_set[x_var].dtype in ["int64", "float64"] and df_set[x_var].nunique() > 10):
            sns.boxplot(x=y_var, y=x_var, data=df_set)
            # Kruskall-Wallis test
            statistic, pvalue = stats.kruskal(df[df[y_var]=='lost'][x_var], df[df[y_var]=='won'][x_var])
            print('Kruskall-Wallis test statistic:', statistic)
            print('Probablity H0 of independent distributions is true:', pvalue)       
            
    elif (df_set[x_var].nunique() <= 10):
            sns.barplot(x=x_var, y=y_var, data=df_set, estimator=np.mean)
            # Pearson chisquare test (only large samples!!!)
            cont_table = pd.crosstab(df[x_var], df[y_var])
            statistic, pvalue, dof, expected = stats.chi2_contingency(cont_table)
            print('Pearson Chi-square test statistic:', statistic)
            print('Probablity H0 of independent distributions is true:', pvalue)

    else:
        "dtype not recognized or too many categories"
        

def cleaning(text):

    import string
    exclude = set(string.punctuation)
    import re
    # remove new line and digits with regular expression
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\d', '', text)
    # remove non-ascii characters
    text = ''.join(character for character in text if ord(character) < 128)
    # remove punctuations
    text = ''.join(character for character in text if character not in exclude)
    # standardize white space
    text = re.sub(r'\s+', '_', text)
    # drop capitalization
    text = text.lower()
    #remove white space
    #text = text.strip()

    return text