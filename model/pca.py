import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA


get_partition = lambda vec, N: np.argpartition(vec, -N)[-N:]

reduce_topK = lambda vec, part: [ 1/len(part) if M in part else 0. for M in range(len(vec)) ]

reduce_eigen = lambda vec: [ 1. if np.argmax(vec) == M else 0. for M in range(len(vec)) ]

def eigenports(rets, num_components=3, reduce=False):
  '''
  reduce dimensionality via PCA
  generate orthonormal investment Universes (principal Eigenportfolios)
  
  rets - returns matrix
  num_components - how many PCA vectors to discover
  reduce - boolean to reduce vector further to top K assets only if desired 
  '''
  eigenvecs = [] 
    
  cols = rets.columns.unique()  
  feat_num = len(cols)

  pca = PCA(n_components=num_components)  
  components = pca.fit_transform(rets[cols].cov())

  for M in range(num_components):
    eigenvec = pd.Series(index=range(feat_num), data=pca.components_[M])
    prt = get_partition(eigenvec, 1).values

    eigenvecs.append(
      reduce_topK(eigenvec, prt)
        if reduce 
        else eigenvec)

  return eigenvecs, pca, components
