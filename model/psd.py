import numpy as np 
import scipy
import scipy.stats as stats 

class FractionalPSD:
    def __init__(self, data):
        self.data = np.array(data)

    def distance_norm(self, target): 
        x1 = torch.tensor([ _x for _x in target ], dtype=torch.float)
        x0 = torch.tensor([ _x for _x in self.data  ], dtype=torch.float)
        
        # Evaluate kernel auto-covariance matrix 
        kernel = gpytorch.kernels.RBFKernel()
        autocov_1 = kernel(x1, x1).evaluate().detach().numpy()
        autocov_0 = kernel(x0, x0).evaluate().detach().numpy()
        self.autocov = autocov_0
        
        # Frobius distance 
        return np.linalg.norm(autocov_1 - autocov_0, 'fro') 
    
''' OLD
class FractionalPSD:
    def __init__(self, data):
        self.data = np.array(data)
        self.data_expon = np.random.exponential(scale=1 /(np.abs(scipy.fft.fft(self.data)) ** 2) )
    
    def wasserstein_score(self, target):

        # Distance score for Power Spectral Density (PSD) TODO
        return stats.wasserstein_distance(np.random.exponential(scale=1 /(np.abs(scipy.fft.fft(target)) ** 2)), 
                                            self.data_expon)
'''
