from sklearn.mixture import BayesianGaussianMixture
import numpy as np

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
        - transition_kernel (2d array): A transition probability matrix for each time step.  
        """
        self.covariances = covariances
        self.transition_kernel = transition_kernel   
        self.means = means

    """
    Sample from the theory as a data generating process.
    """
    def sample(self, num_samples=100): 
        
        # Declare empty states
        choices = np.zeros(num_samples)

        # Sample the posterior distribution states based on transition matrix and multivariate normal 
        if len(self.covariances) > 1: 
            choices = [ np.random.choice(len(self.covariances), 
                                        p=self.transition_kernel[i][int(choices[i-1])]) for i in range(num_samples) ]

        # Get X, and y dim 
        Xy = np.array([ 
                       np.random.multivariate_normal(
                           mean=self.means[state], 
                           cov=self.covariances[state]) for state in map(int, choices) ])
        return Xy


class RiskyForecast:
    """
    A class that represents a RiskyForecast which takes a BayesianGaussianMixture (should be trained) instance as input.
    """
    def __init__(self, corpus: BayesianGaussianMixture):
        """
        Constructor that accepts a BayesianGaussianMixture instance.

        Parameters:
        - corpus (BayesianGaussianMixture): A BayesianGaussianMixture instance.

        """
        assert corpus.means_ is not None
        self.corpus = corpus
     
    def likelihood(self, theory: DGPTheory, num_iter=100):
        """
        Method that returns the time-varying likelihood of the theory given the pre-fitted variational
        inference model, which represents the "riskiness" in a Bayesian sense. It indicates how unlikely  
        the outcomes are it is predicting, and is related to the falsifiability of the theory.

        Parameters:
        - theory (DGPTheory): An instance of the DGPTheory class.

        Returns:
        - likelihood (float): The likelihood of the theory over time. 
        
        """
        sampled_theory = theory.sample(num_iter)
        sample_likelihood = [] 
        
        # Iterate over test data
        for _xy in sampled_theory:
            
            # Compute log likelihood assuming the existing posterior parameters 
            sample_likelihood = np.append(sample_likelihood, 
                                          self.corpus.score(np.array([_xy])) ) 

        return sample_likelihood, sampled_theory
        
