import math 
import time 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.integrate as integrate

from sklearn.model_selection import train_test_split
from .pca import eigenports



dotprod = lambda v1, v2: sum((a * b) for a, b in zip(v1, v2))

length = lambda v: math.sqrt(dotprod(v, v))
 
angle = lambda v1, v2: math.acos(dotprod(v1, v2) / 
                                 (length(v1) * length(v2)))

def step_thru_integral(series, W=28): 
  I=[]
  for X in range( W, len(series) ): 
    I.append( integrate.simps(series[X - W: X], even='first') )
  return I
  
lebesgue_ = lambda series: step_thru_integral(series) 



def relative_performances(vector1, vector2, vector3, returns): 
  ''' relative_performances 
  '''
  w1 = vector1 / np.sum(vector1)
  w2 = vector2 / np.sum(vector2)
  w3 = vector3 / np.sum(vector3)
  n=0
  for coltext in returns.columns:
    returns.rename(columns={coltext:n}, inplace=True)
    n+=1

  wr1 = w1 * returns
  wr2 = w2 * returns
  wr3 = w3 * returns
  
  return wr1.sum(axis=1), wr2.sum(axis=1), wr3.sum(axis=1)



def covariance_regimes(returns, distances=dict(), max_depth=0, test_ratio=0.5, eigenvecs=None): 
  ''' covariance_regimes
  '''
  prior, adhoc = (returns, returns)

  if max_depth > 0:
    prior, adhoc = train_test_split(returns, train_size=test_ratio, test_size=(1.-test_ratio), shuffle=False)
    for K in [prior, adhoc]:
      distances = covariance_regimes(K, distances, max_depth=(max_depth - 1), eigenvecs=eigenports(prior))
  else:
    eigenvecs = eigenports(returns)
    
  if eigenvecs is not None and max_depth == 0:
    
    b1, b2, b3, _ = eigenvecs     
    distances[returns.index.values[0]] = relative_performances(b1, b2, b3, returns)
    
  return distances
