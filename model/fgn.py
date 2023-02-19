import numpy as np
from scipy.linalg import toeplitz

# Autocovariance Function - T time point, exponent 2H (2 * Hurst parameter) 
correlation = lambda T, exp: 1 if T == 0 \
                                else (((T+1)**exp + (T-1)**exp - 2*(T**exp)) / 2)

# Fractional Gaussian Noise Covariance series 
# Resources:
#   Basic Properties of the Multivariate Fractional Brownian Motion 
#   https://hal.science/hal-00497639/document
#
def fractional_gaussian_covariance(n:int, hurst1:float, hurst2:float, corr:float, sigma1:float=1, sigma2:float=1):
    
    # HAL doc page 4 cross covariance function (6)
    # Cross-covariance of the increments
    w = lambda h: corr * abs(h)**(hurst1 + hurst2)
    # w is the weight at i and jth position in the matrix 
    series = w(n-1) - 2*w(n) + w(n+1)
    # Multiply sigmas of the i and jth divided by 2 
    series = sigma1 * sigma2 / 2 * series
    return series

def fractional_gaussian_covariance_series(size:int, hurst1:float, hurst2:float, corr:float, sigma1:float=1, sigma2:float=1):

    series = np.ndarray((2*size, 2*size))   
    series[:size, :size] = toeplitz([ correlation(i, hurst1) for i in range(size) ]) * (sigma1**2)
    series[-size:, -size:] = toeplitz([ correlation(i, hurst2) for i in range(size) ]) * (sigma2**2)

    for i in range(size):
        for j in range(size):
            series[i, size+j] = series[size+j, i] = (fractional_gaussian_covariance(i-j, hurst1, hurst2, corr, sigma1, sigma2))
            
    return series
