'''
The main function and purpose of this module is the 'naive_fit' function, which combines scikit learn's logistic regression (with the 
default settings), along with filling in guesses for the missing cure labels for the censored rows, and there are four different ways of 
doing this by using the 'initialize' parameter.

'''




import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from make_df import create_df, make_inputs
from sklearn.cluster import KMeans
from hardEM_0 import vswap, dist, HEM_labels_fit


 

def naive_fit(censored_inputs, noncensored_inputs, initialize):
    
    '''
    Fit the model using the given censored and noncensored inputs, and return predictions and probabilities.
    
    Parameters
    ----------------------------------------
    
    censored_inputs: A numpy array of shape (n_samples, n_covariates+1), containing the censored rows, each of which begins with a '1', 
    accounting for our intercept term. E.g. array([[1,0.2,0.3], [1,0.5,0.6]]) has two samples and two covariates.
    
    noncensored_inputs: Same as above except these represent the noncensored rows. The number of covariates is the same as above, but we
    could have a different number of samples. 
    
    initialize: Choice of populating the missing cure labels for the censored rows. The options are 'censoring_rate', 
    'use_clustering', 'use_HardEM', or 'fifty_fifty'. If the option 'censoring_rate' is selected, then we initialize by assuming the 
    probability of being cured is the censoring rate, and use this to generate cure labels for the censored rows. Otherwise, if 
    'use_clustering' is selected then a single cluster is created from the noncensored rows, and two clusters are created from the censored 
    rows. By comparing the distance of the two censored cluster centers to the noncensored cluster center, cure labels are assigned to the 
    censored rows. The most important of these is the 'use_HardEM' option which takes as missing labels those generated by the HardEM
    algorithm, which is carried out using the 'HEM_labels_fit' function from from hardEM_0.py. In our implementation of the latter, we 
    chose a regularization of 0.5 and maximum number of iterations of 1000.
    
    
    Returns
    ----------------------------------------
    
    The output is a dictionary containing two keys: (i) 'pred', whose value is the predicted label of the corresponding row, and (ii) 
    'prob', whose value is the corresponding probability of *not* being cured (i.e. label 1).
    
    '''
    
    censored_inputs = censored_inputs[:,1:] #To feed into scikit learn, we omit the unnecessary leading 1's.
    
    noncensored_inputs = noncensored_inputs[:,1:]
    
    n_noncens = len(noncensored_inputs)
    n_cens = len(censored_inputs)
    n_rows = n_noncens+n_cens
    
    total_inputs = np.concatenate((censored_inputs, noncensored_inputs), axis=0)
        
    noncens_labels = np.repeat(1,n_noncens)
    
    if initialize == 'censoring_rate':
    
    
        #Use censoring rate to fill in missing values. 

        p = 1-n_cens/n_rows  # probability of getting cured (label 0!) is the censored rate.
        guess_unknown_labels = np.random.binomial(1, p, n_cens)
        # result of flipping a coin once n_cens times.
        
        labels = np.concatenate((guess_unknown_labels, noncens_labels), axis=None)
        
        clf = LogisticRegression(random_state=0).fit(total_inputs, labels)
    
        nonintercept_weights = np.ndarray.flatten(clf.coef_)
    
        intercept = clf.intercept_
    
        weights = np.concatenate((intercept, nonintercept_weights))
    
        predictions = clf.predict(total_inputs)
    
        prob_not_cured = clf.predict_proba(total_inputs)[:,1]
    
        fit = {'pred':predictions, 'prob':prob_not_cured}
       
        
        return fit


    elif initialize == 'use_clustering':
    #Use clustering to initialize the minimization    
        
        
        kmeans_nc = KMeans(n_clusters=1, random_state=0).fit(noncensored_inputs)

        noncensored_clust_cent = kmeans_nc.cluster_centers_[0]
        
        kmeans_c = KMeans(n_clusters=2, random_state=0).fit(censored_inputs)
        
        censored_clust_cent = kmeans_c.cluster_centers_
        
        cens_cluster1_cent = censored_clust_cent[0]

        cens_cluster2_cent = censored_clust_cent[1]
        
        c_labels = kmeans_c.labels_
        
        if dist(cens_cluster1_cent, noncensored_clust_cent)<dist(cens_cluster2_cent, noncensored_clust_cent):
    
            c_labels = vswap(c_labels)
        
        guess_unknown_labels = c_labels
        
        labels = np.concatenate((guess_unknown_labels, noncens_labels), axis=None)
        
        clf = LogisticRegression(random_state=0).fit(total_inputs, labels)
    
        nonintercept_weights = np.ndarray.flatten(clf.coef_)
    
        intercept = clf.intercept_
    
        weights = np.concatenate((intercept, nonintercept_weights))
    
        predictions = clf.predict(total_inputs)
    
        prob_not_cured = clf.predict_proba(total_inputs)[:,1]
    
        fit = {'pred':predictions, 'prob':prob_not_cured}
       
        
        return fit
    
    elif initialize == 'use_HardEM':
    #Use HardEM algorithm to generate the censored cure labels
    
        guess_unknown_labels = HEM_labels_fit(censored_inputs, noncensored_inputs, 0.5, 1000, 'use_random')
    
        labels = np.concatenate((guess_unknown_labels, noncens_labels), axis=None)
        
        clf = LogisticRegression(random_state=0).fit(total_inputs, labels)
    
        nonintercept_weights = np.ndarray.flatten(clf.coef_)
    
        intercept = clf.intercept_
    
        weights = np.concatenate((intercept, nonintercept_weights))
    
        predictions = clf.predict(total_inputs)
    
        prob_not_cured = clf.predict_proba(total_inputs)[:,1]
    
        fit = {'pred':predictions, 'prob':prob_not_cured}
        
        
        return fit
    
    elif initialize == 'fifty_fifty':
    #Here we assume that each censored row has a 50% chance of being cured.
         

        p = 0.5  # probability of getting cured (label 0!) is the 0.5.
        guess_unknown_labels = np.random.binomial(1, p, n_cens)
        # result of flipping a coin once n_cens times.
        
        labels = np.concatenate((guess_unknown_labels, noncens_labels), axis=None)
        
        clf = LogisticRegression(random_state=0).fit(total_inputs, labels)
    
        nonintercept_weights = np.ndarray.flatten(clf.coef_)
    
        intercept = clf.intercept_
    
        weights = np.concatenate((intercept, nonintercept_weights))
    
        predictions = clf.predict(total_inputs)
    
        prob_not_cured = clf.predict_proba(total_inputs)[:,1]
    
        fit = {'pred':predictions, 'prob':prob_not_cured}
       
        
        return fit
       
    
    
    else: raise ValueError(
        "Need initialize parameter to be chosen as either 'use_clustering', 'censoring_rate', 'use_HardEM', or use 'fifty_fifty'")

        
        
                        
        
        
def get_true_labels(foo, cols):
    
    '''
    Returns:
    
    ---------------------
    
    This function returns a dataframe in which the rows can be partitioned into two disjoint pieces, the top piece being the censored rows
    and the bottom piece the noncensored rows. The reason we need this function is due to the order in which we input our data into 
    scikit-learn's logistic regression, in the 'naive_fit' function above.
    
    Parameters:
    
    ----------------------
    
    foo: A dataframe, which contains censored and noncensored rows.
    
    cols: A list of two strings of the form e.g. ['status', 'cure_label']. These correspond to the censoring indicator and cure label 
    columns of 'foo', each having values either 0 or 1. The order of the strings in the list 'col' is important. In particular, the first 
    string is the censoring indicator and second is the cure label.
      
    
    '''

    censoring_indicator = cols[0]
    cure_label = cols[1]
   
   
    censored = foo[foo[censoring_indicator]==0]
    noncensored = foo[foo[censoring_indicator]==1]
    
    true_cens_labels = censored[cure_label]
    
    true_noncens_labels = noncensored[cure_label]
                                      
    true_labels = np.concatenate((true_cens_labels, true_noncens_labels), axis=0)
                                      
    return true_labels                                  

 



        
        
        
     
