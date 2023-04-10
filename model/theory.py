from sklearn.mixture import BayesianGaussianMixture
from sklearn.covariance import GraphicalLasso, EmpiricalCovariance

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

class Theory:
    
    """
    A class that represents a Theory, which is a simple wrapper for three variables: array of means,
    array of covariance matrices, and transition probability matrix.
    """
    def __init__(self, means, covariances, transition_matrix):
        """
        Constructor that accepts arrays of means, covariance matrices, and a transition probability matrix.

        Parameters:
        - means (array): An array of means.
        - covariances (array): An array of covariance matrices.
        - transition_matrix (array): A transition probability matrix.
        
        E.g. this could be a shrunk set of covariance matrices, means, + transition probabilities 
        See below 
        
        """
        self.covariances = covariances
        self.transition_matrix = transition_matrix
        # TODO implement later 
        self.means = means

    """
    Sample.
    """
    def sample(self, num_samples=100): 
        
        # Declare
        choices = np.zeros(num_samples)
        print(self.transition_matrix)
        
        # Sample the posterior distribution states based on transition matrix and multivariate normal 
        choices = [ np.random.choice(len(self.covariances), 
                                     p=self.transition_matrix[int(choices[i-1])]) for i in range(num_samples) ]
        
        sample0 = [ np.random.multivariate_normal(mean=self.means[state], cov=self.covariances[state])[0] for state in choices ]
        sample1 = [ np.random.multivariate_normal(mean=self.means[state], cov=self.covariances[state])[1] for state in choices ]
        return sample0, sample1


class TheoryMachine:
    """
    A class that represents a TheoryMachine which takes a list of BayesianGaussianMixture instances as input.
    """
    def __init__(self, corpus: BayesianGaussianMixture):
        """
        TODO will be list 
        Constructor that accepts a list of BayesianGaussianMixture instances, each representing a "corpus" or
        a pre-trained model (posterior distribution after fitting).

        Parameters:
        - corpus (BayesianGaussianMixture): A BayesianGaussianMixture instance.

        """
        self.corpus = corpus

    def parameterize(self, theory: Theory):
        """
        Method that accepts an instance of the Theory class and performs parameterization using the
        BayesianGaussianMixture models in the corpus_list.

        Parameters:
        - theory (Theory): An instance of the Theory class.

        """
        # Perform parameterization using BayesianGaussianMixture models in corpus_list
        # Implementation left for user

    def likelihood(self, theory: Theory, num_iter=100):
        """
        Method that returns the cumulative likelihood function of the theory given the pre-fitted corpus models,
        which represents the "riskiness" in a Bayesian sense of the theory. It indicates how unlikely are the outcomes
        it is predicting/covering, and is related to the falsifiability of the theory.

        Parameters:
        - theory (Theory): An instance of the Theory class.

        Returns:
        - likelihood (float): The cumulative likelihood function of the theory.
        
        """
        sample0, sample1 = theory.sample(num_iter)
        sample_likelihood = [] 
        
        # Iterate over test data
        for X, y in zip(sample0, sample1):

            Xy = 0
            # Compute log likelihood assuming the existing posterior parameters 
            # across all states (avg) without assuming which state it is, to get a global
            # score of log-likelihood deviation?
            import scipy
            for m, cov in zip(self.corpus.means_, self.corpus.covariances_): 
                log_likelihood = scipy.stats.multivariate_normal.logpdf(
                    [X, y], mean=m, cov=cov
                )
                Xy += log_likelihood
                
            print(Xy)
            
            sample_likelihood = np.append(sample_likelihood, Xy / len(self.corpus.means_))
            
        import scipy.stats as stats
        print(sample_likelihood)
        return sample_likelihood
            
      
        # Calculate the cumulative likelihood function of the theory given the pre-fitted corpus models
        # Implementation left for user

        # Return the calculated likelihood
        # likelihood = ...
        # return likelihood

    # Other methods and properties of TheoryMachine class
    # ...



# Generate example data
#np.random.seed(0)

import numpy as np

# Transition matrix
transition_matrix = np.array([[0.2, 0.4, 0.4], [0.2, 0.4, 0.4], [0.2, 0.4, 0.4]])

# State means
state_means = np.array([[-1, 0], [2, 5], [-3, -2]])

# State covariance matrices
state_covs = np.array([[[1, 0], [0, 1]], [[2, 1], [1, 2]], [[0.5, 0.2], [0.2, 0.5]]])

# Generate the time series simulation
size=1000
time_series = np.zeros((size, 2))
state = np.random.choice(3)  # Start with a random state

for t in range(size):
    # Update state
    state = np.random.choice(3, p=transition_matrix[state])

    # Generate sample from current state
    sample = np.random.multivariate_normal(mean=state_means[state], cov=state_covs[state])
    
    # Add sample to time series
    time_series[t] = sample

# Final time series arr
#n_samples = 1000
n_features = 2
X = time_series #np.random.randn(n_samples, n_features)

# Estimate empirical covariance matrix
empirical_cov = EmpiricalCovariance().fit(X).covariance_

# Perform covariance shrinkage using Graphical Lasso method
graphical_lasso = GraphicalLasso().fit(X)
shrunk_cov_graphical_lasso = graphical_lasso.covariance_

# Compare estimated covariance matrices
print("Empirical Covariance Matrix:\n", empirical_cov)
print("Shrunk Covariance Matrix (Graphical Lasso):\n", shrunk_cov_graphical_lasso)

# Instantiate mixture model 
num_components=19
dpgmm_model = BayesianGaussianMixture(n_components=num_components, weight_concentration_prior_type='dirichlet_process', n_init=1, max_iter=10000)
states = dpgmm_model.fit_predict(X)

# Initialize the transition matrix with zeros
n_dims = len(set(states))
transition_matrix = np.zeros((num_components, num_components))

# Loop through the array of states and update the transition matrix
for i in range(len(states)):
    current_state = states[i]
    previous_state = states[i-1]
    transition_matrix[previous_state, current_state] += 1

# Normalize the transition matrix and zero out connections to un-used clusters
transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
transition_matrix[np.isnan(transition_matrix)] = 1 / num_components

# Make Theory from the corpus (usually this would be different)
#transition_matrix = [[0.1,0.9],[0.1,0.9]] # overwriting
perturb = 1.25
theory1 = Theory(dpgmm_model.means_, dpgmm_model.covariances_, transition_matrix)
theory2 = Theory(dpgmm_model.means_ * perturb, dpgmm_model.covariances_ * perturb, transition_matrix)
#theory2 = Theory(dpgmm_model.covariances_ * 1.33 , [[0.1,0.9],[0.1,0.9]])

# Instantiate theory machine where dpgmm_model is the corpus 
theory_machine = TheoryMachine(dpgmm_model)

# Test theory 
sample_likelihood1 = theory_machine.likelihood(theory1, num_iter=99)
sample_likelihood2 = theory_machine.likelihood(theory2, num_iter=99)

sdiff1 = np.array(sample_likelihood1) 
sdiff2 = np.array(sample_likelihood2)

sdiff_stabilized1=[ np.mean(sdiff1[:n]) for n in range(10, len(sdiff1)) ]
sdiff_stabilized2=[ np.mean(sdiff2[:n]) for n in range(10, len(sdiff2)) ]

fig,ax=plt.subplots()
axx=ax.twinx()
sns.lineplot(data=sdiff_stabilized1, ax=ax)
sns.lineplot(data=sdiff_stabilized2, ax=ax)
sns.lineplot(data=np.array(sdiff_stabilized2) - np.array(sdiff_stabilized1), linestyle='--', label='diff', ax=axx)

plt.title('bayes risk score')
plt.show()
