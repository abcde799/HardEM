
from make_df import make_inputs  # Custom module here 
from naive import naive_fit  # Custom module here
from scipy.optimize import minimize
import autograd.numpy as np
from autograd import grad
import pandas as pd
from survival_func import mod_prob_density, mod_overall_survival, survival_fit


def nc_survival_fit_weights(censored_inputs, noncensored_inputs, nc, maxiter):
    '''
    Minimize the negative log of MLE to get gamma, the covariate weight vector.
    See Eq 19 in paper.
    Parameters:
    ----------------------------
    censored_inputs: An array consisting of censored inputs of the form [time, prob, covariates].
    E.g. [[1, 0.5, [1.4,2.3,5.2]],...].
    
    noncensored_inputs: Same as above except these represent the noncensored rows. The number of covariates is the same as above, but we
    could have a different number of samples. 
    
    nc: No. of covariates.
    
    maxiter: Maximum number of iterations for numerical solver
    
    
    Returns:
    
    ---------------------------------
    
    Weights: [scaling, shape, gamma], which is flat array where gamma is the covaraite vector weights. 
    ''' 
    n_cens = len(censored_inputs)
    n_noncens = len(noncensored_inputs)
    n_rows = n_cens+n_noncens    
    
    def training_loss(flatparam):
    
      
        arr =  [0.0]*nc #gamma

        param = [flatparam[0], flatparam[1], arr] #[scaling, shape, gamma]
       
        #Training loss is the negative log-likelihood.
        
        known_loss = np.log(np.array(mod_prob_density(noncensored_inputs, param))) #noncensored loss term 
        unknown_loss = np.log(np.array(mod_overall_survival(censored_inputs, param))) #censored loss term
        #reg = np.dot(np.array(arr),np.array(arr))
    
        return -1/n_rows*(np.sum(known_loss)+np.sum(unknown_loss))
        
    training_gradient = grad(training_loss)

    #hess = hessian(training_loss)
    
    length = 2

    b = (0.001,None) #Make sure that both the shape and scaling parameter positive. 
    
    bnds = (b,b)
    
    guess = np.random.uniform(low=0.1, high=0.9, size=length)

    res = minimize(training_loss, guess, method='SLSQP', jac=training_gradient, bounds=bnds, options={'maxiter': maxiter})

    model_weights = res.x
    
    log_likelihood = (-n_rows)*training_loss(model_weights)
    
    #observed_information_matrix = n_rows*hess(model_weights)
    
    #stand_errors = np.sqrt(inv(observed_information_matrix).diagonal()) 
    
    
    return log_likelihood
    
def nc_survival_fit(censored_inputs, noncensored_inputs, nc, maxiter):
    
    ''' 
    Same inputs and outputs as survival_fit_weights. Due to bugs with Autograd, we need to keep running survival_fit until it outputs correctly. 
    '''
    result = None
    while result is None:
        
        try:
        
            result = nc_survival_fit_weights(censored_inputs, noncensored_inputs, nc, 100)
        
        except:
            
            pass
            
        
    return result


if __name__ == '__main__':

    mel = pd.read_csv('melanoma.csv')
    mel['time'] = mel['time']/365.25 
    censoring_indicator = mel['status'] 
    censoring_indicator = censoring_indicator.replace(3, 0)
    censoring_indicator = censoring_indicator.replace(2, 0)
    mel['status'] = censoring_indicator
    
    def prepare_data(mel):
        
        censored_time = mel[mel.status==0].time
        noncensored_time = mel[mel.status==1].time
        covariates = ['sex', 'age', 'thickness', 'ulcer'] #Omit time, year, and status
        mel_covariates = mel[covariates] 
        mel_covariates=(mel_covariates-mel_covariates.mean())/mel_covariates.std() #standardize col. wise.
        mel_covariates['status'] = censoring_indicator
        
        #Extract censored inputs (status label 0) and noncensored inputs 
        #(status label 1) using our own function 
    
        columns = ['status']
        censored_inputs = make_inputs(mel_covariates, 0, columns)  # extract censored
        noncensored_inputs = make_inputs(mel_covariates, 1, columns) # extract noncensored
        fit = naive_fit(censored_inputs, noncensored_inputs, 'use_HardEM')

        y_pred = fit['pred']  # predicted label of the corresponding row
            
        y_scores = fit['prob']  # This is the probability of *not* being cured

        censored_mel = mel_covariates[mel_covariates['status'] == 0]

        noncensored_mel = mel_covariates[mel_covariates['status'] == 1]
        
        censored_mel['time'] = censored_time
        
        noncensored_mel['time'] = noncensored_time
        
        final = pd.concat([censored_mel, noncensored_mel]) 
    
        final['predicted_prob_cured'] = 1-y_scores
        
        final_cov = final[[col for col in final.columns if col not in ['time', 'predicted_prob_cured']]]
        
        cens_cov = final_cov[final_cov.status==0]
    
        noncens_cov = final_cov[final_cov.status==1]
        
        final_time_prob_cens = (final[final.status==0])[['time', 'predicted_prob_cured']]
    
        final_time_prob_noncens = (final[final.status==1])[['time', 'predicted_prob_cured']]
        
        cens_cov = cens_cov.drop(columns=['status'])
        
        print('covariates are: {} in that order'.format(list(cens_cov.columns)))
        
        times_n_probs = final_time_prob_cens.values.tolist()
        
        covs = cens_cov.values.tolist()
        
        z = list(zip(times_n_probs, covs))
        
        censored_inputs = [[arr[0][0], arr[0][1], arr[1]] for arr in z]
        
        noncens_cov = noncens_cov.drop(columns=['status'])
    
        ntimes_n_probs = final_time_prob_noncens.values.tolist()
    
        noncens_covs = noncens_cov.values.tolist()
    
        zn = list(zip(ntimes_n_probs, noncens_covs))
    
        noncensored_inputs = [[arr[0][0], arr[0][1], arr[1]] for arr in zn]
        
        return censored_inputs, noncensored_inputs, final
            
    
    nc_arr = []
    arr = []
    for i in range(100):
        
        censored_inputs, noncensored_inputs, table = prepare_data(mel)
        
        nc_arr.append(nc_survival_fit(censored_inputs, noncensored_inputs, 4, 100))
        
        arr.append(survival_fit(censored_inputs, noncensored_inputs, 0, 100, log_likelihood=True))
    
    dic = {}
    dic['no_covariates_log_likelihood'] = nc_arr
    dic['covariates_log_likelihood'] = arr
    
    df = pd.DataFrame.from_dict(dic) 
    df.to_csv('hyp. test. after 100 runs.csv', index=False)

    
