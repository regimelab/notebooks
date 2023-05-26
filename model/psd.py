import numpy as np 
import scipy
import scipy.stats as stats 

class FractionalPSD:
    def __init__(self, data):
        self.data = np.array(data)
        self.data_expon = np.random.exponential(scale=1 /(np.abs(scipy.fft.fft(self.data)) ** 2) )
    
    def wasserstein_score(self, target):

        # Distance score for Power Spectral Density (PSD) TODO
        return stats.wasserstein_distance(np.random.exponential(scale=1 /(np.abs(scipy.fft.fft(target)) ** 2)), 
                                            self.data_expon)
