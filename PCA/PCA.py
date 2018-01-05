# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

Source:
http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_
       python_for_multivariate_analysis.html
       
This source code from the above location was taken, slightly modified 
and modularized and structured for readability
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stat
import matplotlib.rcsetup as rcsetup
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from IPython.display import display, HTML

DISPLAY_MAX_ROWS = 50  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)

from StatLib import *

#==============================================================================
# Colorization of sys.stderr (standard Python interpreter)
#==============================================================================

def PCA1():
       
       print (rcsetup.all_backends)
       
       data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
       
       data.columns
       # rename column names to be similar to R naming convention
       data.columns = ["V"+str(i) for i in range(1, len(data.columns)+1)]  
       data.V1 = data.V1.astype(float)
       # independent variables data
       X = data.loc[:, "V1":]  
       # dependednt variable data
       Y = data.V1  
       #data
       #print (X)
       
       
       #if you want them stacked vertically 
       #f, (ax1, ax2, ax3) = plt.subplots(1, 3)
       
#==============================================================================
# Scatter plot
#==============================================================================
       pd.tools.plotting.scatter_matrix(data.loc[:, "V2":"V6"], diagonal="hist")
       plt.tight_layout()
       plt.show()
       sns.lmplot("V4", "V5", data, hue="V1", fit_reg=True)
       #ax.xaxis.tick_top()

#==============================================================================
# Profile plot
#==============================================================================
       ax = data[["V2","V3","V4","V5","V6"]].plot()
       plt.figure()
       ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

#==============================================================================
# Summary statistics
#==============================================================================      
      
       '''
       print (X.apply(np.mean))
       print (X.apply(np.std))
       '''

#==============================================================================
 #Extract out just cultivar 2 - for example (same can be done for cultivar 1 and 3)
#==============================================================================       

       '''
       class2data = data[Y==2] 
       print (class2data.loc[:, "V2":].apply(np.mean))
       print (class2data.loc[:, "V2":].apply(np.std))
       '''
       
#==============================================================================
# Within and Between Groups Variance 
#==============================================================================       
       #printMeanAndSdByGroup(X, Y)
       
       '''
       print (calcWithinGroupsVariance(X.V2, Y))
       print (calcBetweenGroupsVariance(X.V2, Y))
       calcSeparations(X, Y)
       print ("Within Group Co-Variance = ", calcWithinGroupsCovariance(X.V8, X.V11, Y))
       print ("Between Group Co-Variance = ", calcBetweenGroupsCovariance(X.V8, X.V11, Y))
       '''

#==============================================================================
# Co-orelation text matrix and the heatMap
#==============================================================================      
       
       corrmat = X.corr()
       print ("\n *****FIRST DATA OUTPUT: Co-orelation matrix*****::\n\n", corrmat)
       plt.figure()
       sns.heatmap(corrmat, vmax=1., square=True)
       ax.xaxis.tick_top()

#==============================================================================
# Most highly co-orelated
#==============================================================================       
       
       cor = stat.pearsonr (X.V2, X.V3)
       print ("\n ***** SECOND DATA OUTPUT *****::\n\n")
       print ("Cor:", cor[0], "\t p-value:", cor[1], "\n")
       print ("\n ***** THIRD DATA OUTPUT *****::\n\n")       
       print (mosthighlycorrelated(X, 10))
          
#==============================================================================
# Standardize before running PCA
#==============================================================================
       
       standardisedX = scale(X)
       standardisedX = pd.DataFrame(standardisedX, index=X.index, columns=X.columns)
       standardisedX.apply(np.mean)
       standardisedX.apply(np.std)
       
#==============================================================================
# Run the PCA process
#==============================================================================
       '''
       PCA Process
       '''
       pca = PCA().fit(standardisedX)
       summary = pca_summary(pca, standardisedX)
       plt.figure()
       screeplot(pca, standardisedX)

#==============================================================================
# First Principal Component
#==============================================================================                    
       print ("\n ***** FIRST PRINCIPAL COMPONENT *****::\n\n")
       print (pca.components_[0])
       print ("Sum of Variances:", np.sum(pca.components_[0]**2))

       #Calculate the values of the first principal component
       print (calcpc(standardisedX, pca.components_[0]))
       #Another way - Calculate the values of the first principal component
       #print (pca.transform(standardisedX)[:, 0])
       
#==============================================================================
# Second Principal Component
#==============================================================================
       print ("\n ***** SECOND PRINCIPAL COMPONENT *****::\n\n")
       print (pca.components_[1])
       print ("Sum of Variances: ", np.sum(pca.components_[1]**2))
       
       #Calculate the values of the second principal component
       print (calcpc(standardisedX, pca.components_[1]))
       #Another way - Calculate the values of the second principal component
       #print (pca.transform(standardisedX)[:, 1])

#==============================================================================
# Scatter Plot for the principal components
#==============================================================================       

       pca_scatter(pca, standardisedX, Y)
       
     
       return

#==============================================================================
# End of Main function
#==============================================================================   


#==============================================================================
# Print PCA_Summary
#==============================================================================

def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
    if out:
        print("\n\n********Importance of components**********:\n")
        display(summary)
        display ("Total SDEV from all PCs: ", np.sum(summary.sdev**2))
    return summary

#==============================================================================
# How many Principal Components are enough?
#==============================================================================

def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show() 

#==============================================================================
# Don't remember this one
#==============================================================================

def calcpc(variables, loadings):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # make a vector to store the component
    pc = np.zeros(numsamples)
    # calculate the value of the component for each sample
    for i in range(numsamples):
        valuei = 0
        for j in range(numvariables):
            valueij = variables.iloc[i, j]
            loadingj = loadings[j]
            valuei = valuei + (valueij * loadingj)
        pc[i] = valuei
    return pc

#==============================================================================
# Scatter plot using Principal components
#==============================================================================

def pca_scatter(pca, standardised_values, classifs):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(list(zip(foo[:, 0], foo[:, 1], classifs)), columns=["PC1", "PC2", "Class"])
    plt.figure()
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=True)
#==============================================================================
# Initial call
#==============================================================================

PCA1()



              
