#This is a Hard EM algorithm intended for a mixture model arising from survival analysis. More precisely, this is for situations when we have a cured subpopulation in our survival analysis dataset, but we do not know who is cured. In order to predict the population survival function, we need to know who is cured and instead of using the usual (soft) EM algorithm to predict who is cured via logistic regression, we use our Hard EM algorithm. The difference between the latter two is that our version assigns actual (hard) labels which indicate whether a patient is cured to the censored rows, while the soft EM assigns a probability distribution instead. Clearly, any patient belonging to a noncensored rows is not cured.

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import autograd.numpy as np
from autograd import grad, jacobian, hessian
import pandas as pd
import autograd.numpy as np
from autograd import grad



def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)


def prob(weights, inputs):
    # Outputs probability of a patient being cured according to logistic model.
    return sigmoid(np.dot(inputs, weights))


def HEM_fit(censored_inputs, noncensored_inputs, C, maxiter):
    
    n_noncens = len(noncensored_inputs)
    n_cens = len(censored_inputs)
    n_rows = n_noncens+n_cens


    def training_loss(param):
        #Training loss is the negative log-likelihood of the MLE.
    
        
        weights, unknownlabels = param[0:len(censored_inputs[0])], param[len(censored_inputs[0]):] 
    
        covariate_weights = weights[1:] #do not regularize intercept term
        reg = C*np.dot(covariate_weights, covariate_weights) #regularization term 
        known_loss = 1-prob(weights, noncensored_inputs) #noncensored loss term 
        unknown_loss = np.log(prob(weights, censored_inputs))*(1-unknownlabels)+np.log(1-prob(weights, censored_inputs))*unknownlabels
        
    
        return reg-1/n_rows*(np.sum(np.log(known_loss))+np.sum(unknown_loss))  

    training_gradient = grad(training_loss)


    def constraint1(param):
        # Taking into account lower bnd for I_i(I_i-1) for the censored rows
        weights, unknownlabels = param[0:len(censored_inputs[0])], param[len(censored_inputs[0]):]
    
        return unknownlabels*(unknownlabels-1)+0.001 #unknownlabels

    def constraint2(param):
        # Taking into account upper bnd for I_i(I_i-1) for the censored rows
        weights, unknownlabels = param[0:len(censored_inputs[0])], param[len(censored_inputs[0]):]
    
        return 0.001-unknownlabels*(unknownlabels-1) #-unknownlabels


    #Set the tolerances/bounds for the above constraints 


    cons1 = {'type': 'ineq', 'fun': constraint1}
    cons2 = {'type': 'ineq', 'fun': constraint2}
    cons = [cons1, cons2]


    #Use censoring rate to initialize the minimization


    p = 1-n_cens/n_rows  # number of trials, probability of each trial
    guess_unknown_labels = np.random.binomial(1, p, n_cens)
    # result of flipping a coin once n_cens times.



    guess_weights = np.random.uniform(-0.5,0.5,len(censored_inputs[0]))
    guess = np.concatenate((guess_weights, guess_unknown_labels), axis=None)

    #options0 = {'maxiter': maxiter}

    #minimizer_kwargs0 = {'method':'SLSQP', 'jac':training_gradient, 'constraints':cons, 'options':options0}

    #res = basinhopping(training_loss, guess, minimizer_kwargs=minimizer_kwargs0)

    res = minimize(training_loss, guess, method='SLSQP', jac=training_gradient, constraints=cons, options={'maxiter': maxiter})

    model_weights = res.x[0:len(censored_inputs[0])]
    unknown = res.x[len(censored_inputs[0]):]
    unknown_cure_labels = np.rint(unknown)
    fit = {'model_weights':model_weights, 'unknown_cure_labels':unknown_cure_labels}
    
    return fit



def HEM_predictions(model_weights, df, covariates):
    
    n_rows = len(df.index)
    
    intercept = np.repeat(1, n_rows)
        
    table = df[covariates].to_numpy()
        
    nonintercept_weights = model_weights[1:]
        
    predictions_float = 1-sigmoid(model_weights[0]*intercept+np.dot(table, nonintercept_weights)) 

    predictions = np.rint(predictions_float)

    return predictions





    






