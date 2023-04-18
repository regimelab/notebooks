import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from sklearn.mixture import BayesianGaussianMixture
from vcdf import make_dataframes
import pyeeg

assetlist = ['SPY']
startdate = '2020-1-2'
enddate = '2023-4-14'

for asset in assetlist:
    _df, _vcdf, _, _, _, _, last_predict = make_dataframes(asset, startdate, enddate, volume_per_bar=1e8)
    _vcdf['close'] = _vcdf['close'].dropna()

X = _vcdf.copy()[['close', 'time_per_bar']].diff().dropna()

# Estimate empirical covariance matrix
empirical_cov = EmpiricalCovariance().fit(X).covariance_

# Instantiate mixture model
num_components = 4
dpgmm_model = BayesianGaussianMixture(n_components=num_components, weight_concentration_prior_type='dirichlet_process', n_init=10, max_iter=100000)
states = dpgmm_model.fit_predict(X)

# Count the frequency of each latent state
state_counts = np.zeros(num_components)
for M in states:
    state_counts[M] += 1

# Initialize the transition matrix with zeros
transition_kernel = []
transition_matrix = np.zeros((num_components, num_components))

# Loop through the array of states and update the transition matrix
import copy

for i in range(len(states)):
    current_state = states[i]
    previous_state = states[i - 1]
    transition_matrix[previous_state, current_state] += 1
    transition_kernel.append(update_transition_matrix(copy.deepcopy(transition_matrix)))


# Function to update transition matrix
def update_transition_matrix(transition_matrix):
    # Normalize the transition matrix and zero out connections to un-used clusters
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
    transition_matrix[np.isnan(transition_matrix)] = 1 / num_components
    return transition_matrix

# Instantiate theory machine where dpgmm_model is the corpus sf
theory_machine = RiskyForecast(dpgmm_model)
theory_base = DGPTheory(dpgmm_model.means_, dpgmm_model.covariances_, transition_kernel)

# Test theory
sample_likelihood1, sample1 = theory_machine.likelihood(theory_base, num_iter=len(states))

# Perturb theory
empirical_cov = EmpiricalCovariance().fit(X).covariance_ 
cov_pi = empirical_cov * ( np.eye(empirical_cov.shape[0]) * 0.5 )

# Test perturbed theory 
theory2 = DGPTheory([[np.mean(X['close']), np.mean(X['time_per_bar'])]], [cov_pi], None)
sample_likelihood2, sample2 = theory_machine.likelihood(theory2, num_iter=len(states))    

fig,ax=plt.subplots()
sns.lineplot(data=X['close'].cumsum(), ax=ax, label='baseline', alpha=0.98, color='darkgrey')
sns.lineplot(x=range(len(X), len(X)+len(sample2)), y=np.cumsum([s[0] for s in sample2]), ax=ax, label='theory', alpha=0.48, color='darkred')

ax.set_xlabel('base data [2020-1-2 through 2023-4-14]')
ax.set_title('SPY Reality + SPY Theory')

plt.grid(None)
plt.show()
fig,ax=plt.subplots()
wlen_roll=40
roll1= [ np.mean(sample_likelihood1[x-wlen_roll:x]) for x in range(wlen_roll, len(sample_likelihood1)) ]
roll2= [ np.mean(sample_likelihood2[x-wlen_roll:x]) for x in range(wlen_roll, len(sample_likelihood2)) ]
sns.lineplot(data=roll1, alpha=0.95, color='darkgrey', ax=ax, label='baseline')
sns.lineplot(data=roll2, alpha=0.4, color='red', ax=ax, label='theory')
print(np.sum(roll1))
print(np.sum(roll2))
plt.title('mean likelihood')
ax.set_xlabel('samples over time')
ax.set_ylabel('mean likelihood')
plt.legend(loc='upper left')
plt.show()
