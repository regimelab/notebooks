import numpy as np
from scipy.linalg import toeplitz

def rho_correlation(n: int, H: float):
    
    param = H*2
    if n == 0:
        return 1 
    else:
        return (((n+1)**param + (n-1)**param - 2*(n**param)) / 2) 

def fractional_gaussian_covariance(n:int, hurst1:float, hurst2:float, corr:float, sigma1:float=1, sigma2:float=1):
    
    w = lambda h: corr * abs(h)**(hurst1 + hurst2)
    series = w(n-1) - 2*w(n) + w(n+1)
    series = sigma1 * sigma2 / 2 * series

    return series

def fractional_gaussian_covariance_series(size:int, hurst1:float, hurst2:float, corr:float, sigma1:float=1, sigma2:float=1):

    series = np.ndarray((2*size, 2*size))   
    series[:size, :size] = toeplitz([ rho_correlation(i, hurst1) for i in range(size) ]) * (sigma1**2)
    series[-size:, -size:] = toeplitz([ rho_correlation(i, hurst2) for i in range(size) ]) * (sigma2**2)

    for i in range(size):
        for j in range(size):
            series[i, size+j] = series[size+j, i] = (fractional_gaussian_covariance(i-j, hurst1, hurst2, corr, sigma1, sigma2))
            
    return series
