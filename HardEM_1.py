'''This is a Hard EM algorithm intended for a mixture model arising from survival analysis. More precisely, this is for situations when we have a cured subpopulation in our survival analysis dataset, but we do not know who is cured. In order to predict the population survival function, we need to know who is cured and instead of using the usual (soft) EM algorithm to predict who is cured via logistic regression, we use our Hard EM algorithm. The difference between the latter two is that our version assigns actual (hard) labels which indicate whether a patient is cured to the censored rows, while the soft EM assigns a probability distribution instead. Clearly, any patient belonging to a noncensored rows is not cured.'''

import numpy as np
from scipy.optimize import minimize
import autograd.numpy as np
from autograd import grad, jacobian, hessian
import pandas as pd
import autograd.numpy as np
from autograd import grad
from sklearn.cluster import KMeans


"""Log of the logistic sigmoid function log(1 / (1 + e ** -x)). This function is borrowed from sklearn."""
    
def log_sigmoid(x):    
    if x > 0:
        return -np.log(1. + np.exp(-x))
    else:
        return x - np.log(1. + np.exp(x))



def swap(x): 
   #This one is used for when the 'use_clustering' option is selected for the initialize parameter.
    if x==1:
        
        return 0
    
    if x==0:
        
        return 1
    
vswap = np.vectorize(swap)    


def dist(x,y):
    #This one is used for when the 'use_clustering' option is selected for the initialize parameter.
    return np.sqrt(np.dot(x-y,x-y))
    


def log_prob(weights, inputs):
    # Outputs log of probability of a patient being cured according to logistic model.
    vlog_sigmoid = np.vectorize(log_sigmoid)
    
    return vlog_sigmoid(np.dot(inputs, weights))

def log_one_minus_prob(weights, inputs):
    # Outputs puts log of 1 minus probability of a patient being cured according to logistic model. I.e. log(1-p(weights dot inputs)).
    return -np.dot(inputs, weights)+log_prob(inputs, weights) #This is a numerically stable expression.


def HEM_fit(censored_inputs, noncensored_inputs, C, maxiter, initialize):
    
    '''This function implements the algorithm by minimizing the training loss. To initialize the minimization, which is done using the constrained SLSQP method, we need to choose an initial guess for both the unknown cure labels for the censored rows, and the covariate weights. If the option 'censoring_rate' is selected, then we initialize by assuming the probability of being cured is the censoring rate, and use this to generate cure labels for the censored rows. Otherwise, if 'use_clustering' is selected then a single cluster is created from the noncensored rows, and two clusters are created from the censored rows. By comparing the distance of the two censored cluster centers to the noncensored cluster center, cure labels are assigned to the censored rows. Furthermore, the covraiate weights are initialized at random from a unifrom distribution.
'''
    n_noncens = len(noncensored_inputs)
    n_cens = len(censored_inputs)
    n_rows = n_noncens+n_cens


    def training_loss(param):
        #Training loss is the negative log-likelihood of the MLE.
    
        
        weights, unknownlabels = param[0:len(censored_inputs[0])], param[len(censored_inputs[0]):] 
    
        covariate_weights = weights[1:] #do not regularize intercept term
        reg = C*np.dot(covariate_weights, covariate_weights) #regularization term 
        log_known_loss = log_one_minus_prob(weights, noncensored_inputs) #log of noncensored loss term 
        log_unknown_loss = log_prob(weights, censored_inputs)*(1-unknownlabels)+log_one_minus_prob(weights, censored_inputs)*unknownlabels
        
    
        return reg-1/n_rows*(np.sum(log_known_loss))+np.sum(log_unknown_loss)  

    training_gradient = grad(training_loss)


    def constraint1(param):
        # Taking into account lower bound for I_i(I_i-1) for the censored rows. Default lower bnd taken as 0.001.
        weights, unknownlabels = param[0:len(censored_inputs[0])], param[len(censored_inputs[0]):]
    
        return unknownlabels*(unknownlabels-1)+0.001 #unknownlabels

    def constraint2(param):
        # Taking into account upper bound for I_i(I_i-1) for the censored rows. Default upper bound taken as 0.001.
        weights, unknownlabels = param[0:len(censored_inputs[0])], param[len(censored_inputs[0]):]
    
        return 0.001-unknownlabels*(unknownlabels-1) #-unknownlabels


    #Set the tolerances/bounds for the above constraints 


    cons1 = {'type': 'ineq', 'fun': constraint1}
    cons2 = {'type': 'ineq', 'fun': constraint2}
    cons = [cons1, cons2]
    
    guess_weights = np.random.uniform(-0.5,0.5,len(censored_inputs[0]))


    if initialize == 'censoring_rate':
    #Use censoring rate to initialize the minimization

        p = 1-n_cens/n_rows  # probability of getting cured (lanubel 0!) is the censored rate.
        guess_unknown_labels = np.random.binomial(1, p, n_cens)
        # result of flipping a coin once n_cens times.

        
        guess = np.concatenate((guess_weights, guess_unknown_labels), axis=None)

        res = minimize(training_loss, guess, method='SLSQP', jac=training_gradient, constraints=cons, options={'maxiter': maxiter})

        model_weights = res.x[0:len(censored_inputs[0])]
        unknown = res.x[len(censored_inputs[0]):]
        unknown_cure_labels = np.rint(unknown)
        fit = {'model_weights':model_weights, 'unknown_cure_labels':unknown_cure_labels}
    
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
        
        guess = np.concatenate((guess_weights, guess_unknown_labels), axis=None)
        
        res = minimize(training_loss, guess, method='SLSQP', jac=training_gradient, constraints=cons, options={'maxiter': maxiter})

        model_weights = res.x[0:len(censored_inputs[0])]
        unknown = res.x[len(censored_inputs[0]):]
        unknown_cure_labels = np.rint(unknown)
        fit = {'model_weights':model_weights, 'unknown_cure_labels':unknown_cure_labels}
    
        return fit
    
    
    
    elif initialize == 'use_random':
    #Iterate many random initializations for unknown labels and take the one giving lowest objective 
    
    
        results = {}
        values =  np.array([])
        for p in np.random.uniform(0.1,0.9,50): 
            
            guess_unknown_labels = np.random.binomial(1, p, n_cens)
            # result of flipping a coin once n_cens times.
        
            guess = np.concatenate((guess_weights, guess_unknown_labels), axis=None)
        
            res = minimize(training_loss, guess, method='SLSQP', jac=training_gradient, constraints=cons, options={'maxiter': maxiter})

            model_weights = res.x[0:len(censored_inputs[0])]
            
            value = res.fun
        
            unknown = res.x[len(censored_inputs[0]):]
            
            unknown_cure_labels = np.rint(unknown)
            
            values = np.concatenate((values, value), axis=None)
            
            results[value] = [model_weights, unknown_cure_labels]
            
            
        maxim = values.max()    
        minim = values.min()
           
    
        optimal_model_weights = (results[minim])[0]
        
        optimal_model_labels = (results[minim])[1]
            
        fit = {'model_weights':optimal_model_weights, 'unknown_cure_labels':optimal_model_labels, 'minvalue': minim, 'maxvalue': maxim}
        
        return fit
            
    
    
    
    
    else: raise ValueError("Need initialize parameter to be chosen as either 'use_clustering' or 'censoring_rate', or 'use_random' ")

        
        



def HEM_predictions(model_weights, df, covariates):
    
    '''Gives a prediction column by rounding 1-p where p is the probablity of being cured. This is because the cure label is 0 and 
    the not cured label is 1. The df refers to the original datarame and covariates is the list of covariates from the dataframe
    and in the same order, and similarly for model_weights.'''
    
    n_rows = len(df.index)
    
    intercept = np.repeat(1, n_rows)
        
    table = df[covariates].to_numpy()
        
    nonintercept_weights = model_weights[1:]
        
    predictions_float = 1-sigmoid(model_weights[0]*intercept+np.dot(table, nonintercept_weights)) #This is prob of not cured!

    predictions = {'pred': np.rint(predictions_float), 'prob': predictions_float}

    return predictions




def HEM_labels_fit(censored_inputs, noncensored_inputs, C, maxiter, initialize):
    
    
    fit0 = HEM_fit(censored_inputs, noncensored_inputs, C, maxiter, initialize)

    unknown_cure_labels = fit0['unknown_cure_labels']
    
    return unknown_cure_labels  





