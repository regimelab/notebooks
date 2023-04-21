from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import scipy 

class DGPTheory:
    
    """
    A class that represents a Theory, which is a simple wrapper for three variables: array of means,
    array of covariance matrices, and transition probability matrix for each time step. 
    """
    def __init__(self, means, covariances, transition_kernel):
        """
        Constructor that accepts arrays of means, covariance matrices, and a time-varying transition 
        probability matrix.

        Parameters:
        - means (array): An array of means.
        - covariances (array): An array of covariance matrices.
        - transition_kernel (3d array): A transition probability matrix for each time step.  
        """
        self.covariances = covariances
        self.transition_kernel = transition_kernel   
        self.means = means

    """
    Sample from the theory as a data generating process.
    """
    def sample(self, num_samples=100): 
        
        # Declare empty states
        states = np.zeros(num_samples)

        # Sample the posterior distribution states based on transition matrix and multivariate normal 
        if len(self.covariances) > 1: 
            states = [ np.random.choice(len(self.covariances), 
                                        p=self.transition_kernel[i][int(states[i-1])]) for i in range(num_samples) ]

        # Get X, and y dim 
        Xy = np.array([ 
                       np.random.multivariate_normal(
                           mean=self.means[state], 
                           cov=self.covariances[state]) for state in map(int, states) ])
        return Xy, states


class RiskyForecast:
    """
    A class that represents a RiskyForecast which takes a BayesianGaussianMixture instance as input.
    """
    def __init__(self, corpus: BayesianGaussianMixture):
        """
        Constructor that accepts a BayesianGaussianMixture instance.

        Parameters:
        - corpus (BayesianGaussianMixture): A BayesianGaussianMixture instance that is trained. 

        """
        # assert that the model is already trained
        assert corpus.means_ is not None 
        self.corpus = corpus
        
    def likelihood_base(self, num_iter=100): 
        """
        Method that returns the time-varying likelihood when the theory is just a sample directly from the BayesianGaussianMixture instance. We would expect a high log likelihood.

        Returns:
        - likelihood (float): The likelihood of the theory over time. 
        """
        return self.likelihood(self.corpus, num_iter=num_iter)

    def likelihood(self, theory, num_iter=100):
        """
        Method that returns the time-varying likelihood of the theory given the pre-fitted variational inference model, which represents the "riskiness" in a Bayesian sense. It indicates how unlikely the outcomes are it is predicting, and is related to the falsifiability of the theory.

        Parameters:
        - theory (DGPTheory): An instance of the DGPTheory class.

        Returns:
        - likelihood (float): The likelihood of the theory over time. 
        
        """
        sampled_theory, _ = theory.sample(num_iter)
        sample_likelihood = [] 
        
        # Iterate over test data
        for _xy in range(1, len(sampled_theory)):

            # Compute log likelihood assuming the posterior parameters 
            sample_likelihood = np.append(sample_likelihood, 
                                          self.corpus.score(np.array(sampled_theory[:_xy])) ) 

        return sample_likelihood, sampled_theory
    
     def likelihood_in_regime(self, theory, regime_index, num_iter=100):
        """
        Method that returns the time-varying likelihood of the theory given the pre-fitted variational inference model,
        which represents the "riskiness" in a Bayesian sense, for a specific regime index.

        Parameters:
        - theory (DGPTheory): An instance of the DGPTheory class.
        - regime_index (int): The index of the regime to calculate the likelihood for.
        - num_iter (int): The number of iterations to sample.

        Returns:
        - likelihood (float): The likelihood of the theory in the given regime.
        """
        sampled_theory, sampled_states = theory.sample(num_iter)
        sub_theory = []
        sub_theory_scores = [] 
        
        # Scorer
        _score_ = lambda theory_data, m, cov: np.mean([           
                scipy.stats.multivariate_normal.logpdf(
                    M, 
                    mean=m, 
                    cov=cov
                )
                for M in theory_data 
            ])
        
        # Iterate over test data 
        for _xy, st in zip(sampled_theory, sampled_states):

            # Compute log likelihood assuming the posterior parameters of specific latent state
            m = self.corpus.means_[regime_index]
            cov = self.corpus.covariances_[regime_index]
                
            # Add data and score  
            sub_theory.append([_xy[0],
                            _xy[1]])
            if len(sub_theory) > 1: sub_theory_scores.append(_score_(sub_theory, m, cov))

        return sub_theory_scores, sampled_theory
