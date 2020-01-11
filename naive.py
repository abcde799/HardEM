import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score




def naive_fit(censored_inputs, noncensored_inputs, foo):
    
    
    n_noncens = len(noncensored_inputs)
    n_cens = len(censored_inputs)
    n_rows = n_noncens+n_cens
    
    cens = foo[foo.censoring_indicator==0]
    
    noncens = foo[foo.censoring_indicator==1]
    
    true_labels_cens = cens.cure_label
    
    true_labels_noncens = noncens.cure_label
    
    true_labels = pd.concat([true_labels_cens, true_labels_noncens])
    
    #Use censoring rate to fill in missing values. 

    p = 1-n_cens/n_rows  # number of trials, probability of each trial
    guess_unknown_labels = np.random.binomial(1, p, n_cens)
    # result of flipping a coin once n_cens times.

        
    total_inputs = np.concatenate((censored_inputs, noncensored_inputs), axis=0)
        
    noncens_labels = np.repeat(1,n_noncens)
        
    labels = np.concatenate((guess_unknown_labels, noncens_labels), axis=None)
        
    clf = LogisticRegression(random_state=0).fit(total_inputs, labels)
        
        
    y_pred = clf.predict(total_inputs)
        
    y_prob = clf.predict_proba(total_inputs)
        
    y_true = true_labels
        
        
        
    accuracy = accuracy_score(y_true, y_pred)
       
        
    auc = roc_auc_score(y_true, y_prob)
        
    return [accuracy, auc]

        
        
        
     
