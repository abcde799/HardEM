


from scipy.optimize import minimize
import autograd.numpy as np
from autograd import grad, hessian
import pandas as pd
from autograd import grad
from numpy.linalg import inv


def covariate_exp(covariate_vector, gamma):
    '''Takes two vectors and returns a scalar. The vectors are in the form of a list of equal length. 
    '''
    dot_prod = np.dot(np.array(gamma), np.array(covariate_vector))
    
    return np.exp(dot_prod)
    
def susc_survival(time, covariate_vector, scaling, shape, gamma):
    '''This is the survival function of the susceptible individual. It is Eq 16 from paper.
    Parameters:
    ------------------------------------
    time: Positive scalar; either a float or an integer. 
    
    covariate_vector: A list. 
    
    gamma: A list of length matching 'covariate_vector'.
    
    scaling: positive float; comes from Weibull distribution.
    
    shape: positive float; comes from Weibull distribution.
    
    Returns:
    -----------------------------------
    
    Survival function of susceptible individual at given parameters assuming a Weibull distribution
    for the proportional hazards model with the baseline hazard function having given shape and scale.
    
    '''
    
    arg = (-(time/scaling)**shape)*(covariate_exp(covariate_vector, gamma))
    
    assert np.exp(arg)>=0, "arg is {}. Output of 'susc_survival' is not a nonnegative number".format(arg)

       
    
    #"Output of 'susc_survival' is not a nonnegative number."
    
    
    return np.exp(arg)
    
def overall_survival(time, prob, covariate_vector, scaling, shape, gamma):
    '''Same parameters as 'susc_survival' function above but in addition has:
    prob: Estimated probabilities (to be returned by HardEM); float type between 0 and 1.
    It returns the overall survival function of a (not necessarily susc.) individual. 
    This is Eq 17 from paper. 
    '''
    
    out = prob+(1-prob)*susc_survival(time, covariate_vector, scaling, shape, gamma)
    
    assert 0<=out<=1,"Output of 'overall_survival' is not in [0,1]."
    
    return out
    
        
def prob_density(time, prob, covariate_vector, scaling, shape, gamma):
    '''
    Same parameters as above and returns overall prob density for time of event. This is Eq 18
    from the paper.
    '''
    
    def time_slice(time_param):
        return overall_survival(time_param, prob, covariate_vector, scaling, shape, gamma)
    
    out = -grad(time_slice)(float(time))
    
    assert time>0, 'time<=0' 
    assert (0<=prob<=1), 'prob out of [0,1]' 
    assert scaling>0, 'scaling not positive'
    assert shape>0, 'shape not positive'
    
   
    
    return out
    
def mod_prob_density(Array, param):

    '''
    This is a modification of the 'prob_density' function to be used in the training loss below.
    
    Parameters:
    
    --------------------------
    
    Array: An array of arrays the form [[time, prob, covariate_vector],...]. Here covariate_vector is an array and 
    the others are floats. 
    
    param: An array of the form [scaling, shape, gamma]. Here gamma is an array and the others are floats. 

    Returns:
    
    ---------------------------

    A list of the prob_density function applied to each array in Array with respect to param.    
    '''
    
    out = [prob_density(arr[0], arr[1], arr[2], param[0], param[1], param[2]) for arr in Array]
    
    assert isinstance(out, list),"Output of 'mod_prob_density' is not a list."
    
    return out
    
def mod_overall_survival(Array, param):
    
    '''
    Same modification as mod_prob_density function but for the overall_survival function.
    '''
    
    out = [overall_survival(arr[0], arr[1], arr[2], param[0], param[1], param[2]) for arr in Array]
    
    assert isinstance(out, list),"Output of 'mod_prob_density' is not a list."
    
    return out
    
def survival_fit_weights(censored_inputs, noncensored_inputs, C, maxiter):
    '''
    Minimize the negative log of MLE to get gamma, the covariate weight vector. See Eq 19 in paper.
    
    Parameters:
    
    ----------------------------
    
    censored_inputs: An array consisting of censored inputs of the form [time, prob, covariates].
    E.g. [[1, 0.5, [1.4,2.3,5.2]],...].
    
    noncensored_inputs: Same as above except these represent the noncensored rows. The number of covariates is the same as above, but we
    could have a different number of samples. 
    
    maxiter: Maximum number of iterations for numerical solver
    
    C: Positive float giving the strength of L^2 regularization parameter.
    
    Returns:
    
    ---------------------------------
    
    Weights: [scaling, shape, gamma], which is flat array where gamma is the covaraite vector weights. 
    ''' 
    n_cens = len(censored_inputs)
    n_noncens = len(noncensored_inputs)
    n_rows = n_cens+n_noncens    
    
    def training_loss(flatparam):
    
      
        arr = flatparam[2:] #gamma

        param = [flatparam[0], flatparam[1], arr] #[scaling, shape, gamma]
       
        #Training loss is the negative log-likelihood.
        
        known_loss = np.log(np.array(mod_prob_density(noncensored_inputs, param))) #noncensored loss term 
        unknown_loss = np.log(np.array(mod_overall_survival(censored_inputs, param))) #censored loss term
        reg = np.dot(np.array(arr),np.array(arr))
    
        return C*reg-1/n_rows*(np.sum(known_loss)+np.sum(unknown_loss))
        
    training_gradient = grad(training_loss)

    hess = hessian(training_loss)
    
    length = len((censored_inputs[0])[2])+2 

    b = (0.001,None) #Make sure that both the shape and scaling parameter positive. 
    
    bnds = (b,b)+tuple((None,None) for x in range(length-2)) #The covariate vector components do not need to be positive. 
    
    guess = np.random.uniform(low=0.1, high=0.9, size=length)

    res = minimize(training_loss, guess, method='SLSQP', jac=training_gradient, bounds=bnds, options={'maxiter': maxiter})

    model_weights = res.x
    
    observed_information_matrix = n_rows*hess(model_weights)
    
    stand_errors = np.sqrt(inv(observed_information_matrix).diagonal()) 
    
    
    return model_weights, stand_errors
    
def survival_fit(censored_inputs, noncensored_inputs, C, maxiter, standard_errors=False):
    
    ''' 
    Same inputs and outputs as survival_fit_weights. Due to bugs with Autograd, we need to keep running survival_fit until it outputs correctly. 
    '''
    result = None
    while result is None:
        
        try:
        
            result = survival_fit_weights(censored_inputs, noncensored_inputs, C, 100)
        
        except:
            
            pass
            
    if standard_errors:
        
        return result    
    
    else: 
        
        return result[0]


        
    
        
            








