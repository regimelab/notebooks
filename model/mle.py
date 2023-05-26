     
import torch
import gpytorch
import pyeeg

class FractionalMLE:
    def __init__(self, data):
        self.data = np.array(data)
        self.data_expon = self.expon(data, var=np.var(data))
        
    # Eigenvalues for power spectrum analysis
    def expon(self, x, var):  
            
        # Fit kernel
        kernel = gpytorch.kernels.RBFKernel(lengthscale=0.1)
        kernel_autocov = kernel(torch.tensor(x).unsqueeze(1)).evaluate()

        # Compute the auto-covariance matrix using the RBFKernel
        auto_cov = (kernel_autocov * var).detach().numpy()
            
        # Compute eigenvalues of the auto-correlation matrix  
        eigenvalues = np.linalg.eigvalsh(auto_cov)

        # Generate samples from the exponential distribution for eigenvalues over threshold
        samples = np.random.exponential(scale=1/np.clip(eigenvalues, a_min=1e-20, a_max=None))
        return samples     
    
    def wasserstein_score(self, target):
      
        # Distance score 
        return stats.wasserstein_distance(self.expon(target, np.var(target)), self.data_expon)
