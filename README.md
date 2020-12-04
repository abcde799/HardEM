This is an implementation of a 'Hard EM' algorithm designed for datasets coming from survival analysis. It takes into account a latent cured subpopulation, and assigns actual (hard) cure labels to censored individuals, as well as a probability of being cured to each sample. Non-censored individuals are not cured, whereas censored individuals consist of a mix of cured and not cured individuals. This is an example of learning from positive and unlabeled data (PU learning). This repo has two main features: 

(I) A function to assign cure labels to censored individuals from a survival dataset, as well as the probability of being cured to each sample. A demo of this can be found in the notebook 'demo_hard_em.ipynb'. It should be noted that these methods can be applied to PU learning datasets beyond survival data. 

(II) A function to compute weights as well as standard errors of these weights, using the probabilities outputted in (I) above. These weights are the parameters needed to compute the overall survival function, based on maximum likelihood estimation. A demo of this is in the notebook 'demo_survival_fit.ipynb'.



This is joint work of Nemanja Kosovalic (formerly University of South Alabama) with Sandip Barui - IIM Kozhikode. For any questions please contact Nemanja Kosovalic at n.kosovalic@gmail.com.
