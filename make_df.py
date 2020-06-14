'''
This module consists of functions which generate testing datasets with specified covarites, as well as censored and cured proportions. The 
covariates are drawn from user specified normal distributions. The datasets are intended for testing a hard EM algorithm arising from 
survival analysis in which the cured population is not known.

'''




import numpy as np
import pandas as pd
from hardEM_0 import HEM_predictions



def ranarray(mu, sigma, length):
    
    '''generate a random numpy array with mean mu and standard deviation sigma, from a normal dist of spec. length'''
 
    s = np.random.normal(mu, sigma, length)
    
    return s


         


def add_covariate(df, mu, sigma, n_patients, string):
    
    '''Add a covariate to a dataframe coming from a normal distribution with the given parameters and n_patients rows.'''
    
    df[string] = ranarray(mu, sigma, n_patients)
    
    return df




def make_cov_df(covariates, dist, n_patients):
    
    '''Generate dataframe of covariates drawn from the relevant normal distributions (dist), and having n_patients no. of patients.
    Covariates is a list of strings for the names, and dist is a list of lists each having two components, the first being mu and the 
    second being sigma. E.g. covariates = ['x1', 'x2', 'x3'] and dist = [[3.5, 0.7], [5.1, 2.1], [4.75, 0.9]].'''
    
    new_d = pd.DataFrame()

    for i in range(len(covariates)):
        
        mu = dist[i][0]
        
        sigma = dist[i][1]

        new_d = add_covariate(new_d, mu, sigma, n_patients, covariates[i])
   
    return new_d
   


def add_cure_label(test_model_weights, df, covariates):
    
    '''Uses function from hardEM_0 module to compute a probability rule for being cured, according to test weights.'''
    
    cure_label_dic = HEM_predictions(test_model_weights, df, covariates)
    
    cure_label = cure_label_dic['pred']
    
    df['cure_label'] = cure_label
    
    return df

#Noscar:
def add_nscar_cens_indicator(nscar_weights, df, covariates):
    
    '''Uses function from hardEM_0 module to compute a probability rule for being censored according to nscar_weights, which 
    includes a weight for the intercept term. nscar stands for 'not selected completely at random'. This column is a helper column
    and will not be included in the final dataframe.'''
    
    nscar_dic = HEM_predictions(nscar_weights, df, covariates)
    
    nscar_cens_label = nscar_dic['pred']
    
    df['nscar_cens_label'] = nscar_cens_label
    
    return df


def add_intercept(df):
    
    '''Adds intercept to the left end of dataframe'''
    
    new_col = np.repeat(1,len(df.index))
    
    df['int'] = new_col
    
    return df

def censoring_indicator(cure_label, prob_cens_given_not_cured):
    
    '''Adds censoring labels. If you are cured you must be censored and if you are not cured, then you will
    be censored according to given probability. Note that censored is labeled as 0 and not 1.'''
    
    if cure_label==0:
        
        return 0
    
    elif cure_label==1:
        
        return 1-np.random.binomial(1, prob_cens_given_not_cured)


    
def add_censoring_indicator(df, prob_cens_given_not_cured):
    
    '''Adds censoring indicator column to datadrame.'''
       
    cens_ind_arr = lambda x : censoring_indicator(x, prob_cens_given_not_cured)
    
    df['censoring_indicator'] = df.cure_label.apply(cens_ind_arr)
    
    return df 

#Noscar:

def aux(col1,col2):
    
    '''Helper function used for adding no scar censoring indicator column. If person is cured, then they must be censored, otherwise
    if they are not cured, then they are censored according to probability coming from the relevant column.'''
    
    if col1==0:
        return 0
    else:
        return col2

#Noscar:

def add_nscar_censoring_indicator(df):
    
    '''Adds no scar censoring indicator column to datadrame, using aux helper function.'''
    
    df['nscar_censoring_indicator'] = df.apply(lambda x: aux(x.cure_label, x.nscar_cens_label), axis=1)
       
    
    
    return df    
    


def create_df(covariates, dist, n_patients, test_model_weights, prob_cens_given_not_cured):
    
    '''Puts together the above functions to make a dataframe having the prescribed inputs.'''
    
    foo = make_cov_df(covariates, dist, n_patients)
    
    foo = add_cure_label(test_model_weights, foo, covariates)
    
    foo = add_intercept(foo)
    
    foo = add_censoring_indicator(foo, prob_cens_given_not_cured)
    
    return foo
    
    
#Noscar:

def create_nscar_df(covariates, dist, n_patients, test_model_weights, nscar_weights):
    
    '''Puts together the above functions to make a dataframe having the prescribed inputs, under no scar labelling 
    assumption.'''
    
    foo = make_cov_df(covariates, dist, n_patients)
    
    foo = add_cure_label(test_model_weights, foo, covariates)
    
    foo = add_intercept(foo)
    
    foo = add_nscar_cens_indicator(nscar_weights, foo, covariates)
    
    foo = add_nscar_censoring_indicator(foo)
    
    return foo[['x1', 'x2', 'x3', 'cure_label', 'int', 'nscar_censoring_indicator']]


def make_inputs(df, indicator, cols):
    
    '''Creates numpy array of rows of either censored (0) or noncensored rows (1) depending on indicator, by extracting rows whose
    censoring indicator is zero and rows whose censoring indicator is 1.
    
    
    Parameters:
    
    ----------------------------------------
    
    df: A dataframe of shape (n_samples, n_covariates)
    
    indicator: Either 0 meaning censored, or 1 meaning noncensored.
    
    cols: The list of columns to be dropped. Usually cols = ['censoring_indicator', 'cure_label']
    
    Returns:
    
    
    -----------------------------------------
    
    The output is a numpy array of rows which are either the censored rows, in case the indicator parameter was selected as 0, or 
    otherwise the noncensored rows.
    
    '''
    
    first = cols[0]
    
    new_df = df[df[first]==indicator]
    
    drop_labels = new_df.drop(columns=cols)
    
    add_intercept(drop_labels)
    
    newcols = list(drop_labels.columns)
    newcols = [newcols[-1]] + newcols[:-1]
    drop_labels = drop_labels[newcols]
    
    inputs = drop_labels.to_numpy()
    
    return inputs








