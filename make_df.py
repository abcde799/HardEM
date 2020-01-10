#This module consists of functions which together genrate testing datasets with specified covarites, as well as censored and cured proportions. The covariates are drawn from user specified normal distributions. The datasets are intended for testing a hard EM algorithm arising from survival analysis in which the cured population is not known. 




import numpy as np
import pandas as pd




def ranarray(mu, sigma, length):
    
    '''generate a random numpy array with mean mu and standard deviation sigma, from a normal dist of spec. length'''
 
    s = np.random.normal(mu, sigma, length)
    
    return s


          



def censoring_indicator(cens_prop, n_patients):
    
    '''Returns a numpy array having n_patients entries whose censored proportion (label 0) is cens_prop'''
    
    p_0 = 1-cens_prop
    
    return np.random.binomial(1, p_0, n_patients)




 

def cure_label(delta, cens_prop, cured_prop):
    
    '''If delta_i = 1 take cure_i = 1 and if delta_i = 0, take cure_i = Bin(1-cured_prop/cens_prop).'''
    
    if delta == 1:
        
        return 1
    
    else:
        
        return np.random.binomial(1, 1-cured_prop/cens_prop)   





def cured_cens_df(cens_prop, cured_prop, n_patients):
    
    '''Create a pandas dataframe with censoring indicator column and cured label column having n_patients rows and with the relevant
    indicated cenored proportion and cured proportion.'''

    new_df = pd.DataFrame({'censoring_indicator': censoring_indicator(cens_prop, n_patients) })
    
    cure_label_arr = lambda x : cure_label(x, cens_prop, cured_prop)
    
    new_df['cure_label'] = new_df.censoring_indicator.apply(cure_label_arr)
    
    return new_df





def add_covariate(df, mu, sigma, n_patients, string):
    
    '''Add a covariate to a dataframe coming from a normal distribution with the given parameters and n_patient rows.'''
    
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
   


def create_df(covariates, dist, cens_prop, cured_prop, n_patients):

    '''Make a dataframe by specifying list of strings for covariate names, list of lists each containing mu and sigma for 
    corresponding covariate column, a column indicating who is censored at the rate of cens_prop from total dataset, and 
    a column containing cured labels at the cured_prop rate. Data set has n_patients rows'''

    
    cov_df = make_cov_df(covariates, dist, n_patients)
    cured_cens_df_ = cured_cens_df(cens_prop, cured_prop, n_patients)
    new_df = pd.concat([cov_df, cured_cens_df_], axis=1)
                         
    return new_df


def add_intercept(df):
    
    '''Adds intercept to the left end of dataframe'''
    
    new_col = np.repeat(1,len(df.index))
    
    df['int'] = new_col
    
    return df


def make_inputs(df, indicator, cols):
    
    '''Creates numpy array of rows of either censored (0) or noncensored rows (1) depending on indicator. cols is the
    list of columns to be dropped. Usually cols = ['censoring_indicator', 'cure_label']'''
    
  
    
    first = cols[0]
    
    new_df = df[df[first]==indicator]
    
    drop_labels = new_df.drop(columns=cols)
    
    add_intercept(drop_labels)
    
    newcols = list(drop_labels.columns)
    newcols = [newcols[-1]] + newcols[:-1]
    drop_labels = drop_labels[newcols]
    
    inputs = drop_labels.to_numpy()
    
    return inputs



    
#Blah blah blah    





