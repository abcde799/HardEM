import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from make_df import create_df, make_inputs
from sklearn.cluster import KMeans
from hardEM_0 import vswap, dist, HEM_labels_fit


 

def naive_fit(censored_inputs, noncensored_inputs, initialize):
    
    #Note that this model gives a probability of *not* being cured, so label is 1.
    
    censored_inputs = censored_inputs[:,1:]
    
    noncensored_inputs = noncensored_inputs[:,1:]
    
    n_noncens = len(noncensored_inputs)
    n_cens = len(censored_inputs)
    n_rows = n_noncens+n_cens
    
    total_inputs = np.concatenate((censored_inputs, noncensored_inputs), axis=0)
        
    noncens_labels = np.repeat(1,n_noncens)
    
    if initialize == 'censoring_rate':
    #Use censoring rate to initialize the minimization
    
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
    #Use censoring rate to initialize the minimization
    
        #Use censoring rate to fill in missing values. 

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
       
    
    
    else: raise ValueError("Need initialize parameter to be chosen as either 'use_clustering', 'censoring_rate', 'use_HardEM', or use 'fifty_fifty'")

        
        
                        
        
        
def get_true_labels(foo, cols):
    
    '''cols is the censoring indicator, cure label columns each having values either 0 or 1, and is specified as a list containing two            strings. E.g. ['status', 'curelabel'] in this order. So first is censoring indicator and second is cure label.'''

    censoring_indicator = cols[0]
    cure_label = cols[1]
   
   
    censored = foo[foo[censoring_indicator]==0]
    noncensored = foo[foo[censoring_indicator]==1]
    
    true_cens_labels = censored[cure_label]
    
    true_noncens_labels = noncensored[cure_label]
                                      
    true_labels = np.concatenate((true_cens_labels, true_noncens_labels), axis=0)
                                      
    return true_labels                                  

 



        
        
        
     
