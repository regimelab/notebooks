from sklearn.mixture import BayesianGaussianMixture
import numpy as np


class DGPTheory:
    
    """
    A class that represents a Theory, which is a simple wrapper for three variables: array of means, array of covariance matrices, and transition probability matrix.
    """
    def __init__(self, means, covariances, transition_matrix):
        """
        Constructor that accepts arrays of means, covariance matrices, and a transition probability matrix.

        Parameters:
        - means (array): An array of means.
        - covariances (array): An array of covariance matrices.
        - transition_matrix (array): A transition probability matrix.  
        """
        self.covariances = covariances
        self.transition_matrix = transition_matrix   
        self.means = means

    """
    Sample from the theory as a data generating process.
    """
    def sample(self, num_samples=100): 
        
        # Declare empty states
        choices = np.zeros(num_samples)

        # Sample the posterior distribution states based on transition matrix and multivariate normal 
        choices = [ np.random.choice(len(self.covariances), 
                                     p=self.transition_matrix[int(choices[i-1])]) for i in range(num_samples) ]
        
        # Get X, and y dim 
        Xy = np.array([ 
                       np.random.multivariate_normal(
                           mean=self.means[state], 
                           cov=self.covariances[state]) for state in choices ])
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
        self.corpus = corpus

    def likelihood(self, theory: DGPTheory, num_iter=100):
        """
         Returns the time-varying likelihood function of the theory given the pre-fitted variational inference model, which represents the "riskiness" in a Bayesian sense. It indicates how unlikely are the outcomes it is predicting, and is ultimately related to the falsifiability of the theory.

        Parameters:
        - theory (DGPTheory): An instance of the DGPTheory class.

        Returns:
        - likelihood (float): The likelihood of the theory over time. 
        
        """
        Xy = theory.sample(num_iter)
        return self.corpus.score_samples(Xy)
