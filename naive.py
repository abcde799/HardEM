import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from make_df import create_df, make_inputs




def naive_fit(censored_inputs, noncensored_inputs):
    
    #Note that this model gives a probability of *not* being cured, so label is 1.
    
    censored_inputs = censored_inputs[:,1:]
    
    noncensored_inputs = noncensored_inputs[:,1:]
    
    n_noncens = len(noncensored_inputs)
    n_cens = len(censored_inputs)
    n_rows = n_noncens+n_cens
    
    
    
    #Use censoring rate to fill in missing values. 

    p = 1-n_cens/n_rows  # probability of getting cured (label 0!) is the censored rate.
    guess_unknown_labels = np.random.binomial(1, p, n_cens)
    # result of flipping a coin once n_cens times.

        
    total_inputs = np.concatenate((censored_inputs, noncensored_inputs), axis=0)
        
    noncens_labels = np.repeat(1,n_noncens)
        
    labels = np.concatenate((guess_unknown_labels, noncens_labels), axis=None)
        
    clf = LogisticRegression(random_state=0).fit(total_inputs, labels)
    
    nonintercept_weights = np.ndarray.flatten(clf.coef_)
    
    intercept = clf.intercept_
    
    weights = np.concatenate((intercept, nonintercept_weights))
    
    predictions = clf.predict(total_inputs)
    
    prob = clf.predict_proba(total_inputs)
    
    prob_not_cured = np.ndarray.flatten(prob)[1000:]
    
    
    fit = {'predictions':predictions, 'prob_not_cured':prob_not_cured}
       
        
    return fit


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

 
#def make_inputs(df, indicator, cols):
    
   
  
    
  #  first = cols[0]
   
 #   new_df = df[df[first]==indicator]
    
  #  drop_labels = new_df.drop(columns=cols) 


        
        
        
     
