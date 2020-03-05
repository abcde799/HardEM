

This is an implementation of a 'Hard EM' algorithm desinged from datasets coming from survival analysis. It takes into account a latent 
cured subpopulation, and assigns actual (hard) cure labels to censored indivuduals, whereas noncensored individuals are thought of as not
cured, for obvious reasons. The algorithm is implemented using the using Sequential Least Squares Programming (SLSQP) algorithm from SciPy,
and is contained in the module 'HardEM_0.py'. The input is a collection of rows containing covariates (predictors) for the censored
individuals, as well as covariates for the noncensored individuals, and the output is the missing cure labels for the censored individuals.
In the file 'naive.py' this output is fed into ordinary logistic regression, by selecting the option 'use_HardEM', in the 'naive_fit" 
function.

The file 'make_df.py' is used to generate data sets consisting of censored and noncensored individuals for which we know all of the cure 
labels, so that we can test our algorithm against simpler alternatives such as e.g. populating missing cure labels for censored individuals
by guessing a 50% probability of being cured, etc. This is done by removing the known cure labels and comparing the outputs of both 
algorithms. The notebook 'test_algorithm_scrap.ipynb' contains the results of these tests and shows our algorithm performs better using
both accuracy and auc scores as metrics.

Finally, the notebook 'demo_hard_em.ipynb' contains a demo showing how to use our algorithm for a well known real life melanoma dataset. 
It should be noted our algorithm is used to populate the missing cure labels for the censored individuals, which is then fed into classical
logistic regression using Scikit-learn. 

This work is the result of a collaboration with Nemanja Kosovalic and Sandip Barui from the Department of Mathematics and Statistics
at the University of South Alabama. For questions please contact Nemanja Kosovalic at n.kosovalic@gmail.com
