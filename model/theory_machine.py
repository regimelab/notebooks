import pandas as pd
import numpy as np
from vcdf import make_dataframes
from sklearn.covariance import EmpiricalCovariance
from sklearn.mixture import BayesianGaussianMixture
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from risky_forecast import RiskyForecast, DGPTheory

# Define input variables
assetlist = ['SPY']
featurelist = ['time_per_bar', 'close_abs']
n_estimators = 1000
type_of_sample = 'minutes'
startdate = '2020-1-2'
enddate = '2023-4-19'

# Define a function to digitize a column using percentile bins
digitize_col = lambda N: np.digitize(N, [np.percentile(N, 20), np.percentile(N, 40), np.percentile(N, 60), np.percentile(N, 80)])

# Initialize some variables
df = pd.DataFrame()
vcdf = pd.DataFrame()
last_predict = None

# Loop through assets and create dataframes
for asset in assetlist:
    _df, _vcdf, _, _, _, _, last_predict = make_dataframes(asset, startdate, enddate, volume_per_bar=1e8)
    _vcdf['close'] = _vcdf['close'].dropna()

# Compute X dataframe from _vcdf dataframe
X = _vcdf.copy()[['close', 'time_per_bar']].diff().dropna()

# Estimate empirical covariance matrix
empirical_cov = EmpiricalCovariance().fit(X).covariance_

# Instantiate mixture model 
num_components = 4
dpgmm_model = BayesianGaussianMixture(n_components=num_components, weight_concentration_prior_type='dirichlet_process', n_init=10, max_iter=100000)

# Fit model and predict latent states
states = dpgmm_model.fit_predict(X)

# Count the frequency of each latent state 
state_counts = np.zeros(num_components)
for M in states:
    state_counts[M] += 1 

# Initialize the transition matrix with zeros
transition_kernel = []
transition_matrix = np.zeros((num_components, num_components))

# Define function to update the transition matrix 
def update_transition_matrix(transition_matrix):    
    # Normalize the transition matrix and zero out connections to un-used clusters
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
    transition_matrix[np.isnan(transition_matrix)] = 1 / num_components
    return transition_matrix

# Loop through the array of states and update the transition matrix
for i in range(1, len(states)):
    current_state = states[i]
    previous_state = states[i-1]
    transition_matrix[previous_state, current_state] += 1
    transition_kernel.append(update_transition_matrix(copy.deepcopy(transition_matrix)))

# Instantiate risky forecast model with the trained mixture model 
theory_machine = RiskyForecast(dpgmm_model)

# Compute likelihood of data under the model for one regime
regime = 0
sample_likelihood1, sample1 = theory_machine.likelihood_in_regime(theory_machine.corpus, regime, num_iter=int(len(states)))

# Compute likelihood of data under the model for the same regime but with a modified covariance matrix
pi = 1.314
cov_pi = empirical_cov + (np.eye(empirical_cov.shape[0]) * (empirical_cov[0][0] * pi))
theory2 = DGPTheory(dpgmm_model.means_, cov_pi, transition_kernel)
sample_likelihood2, sample2 = theory_machine.likelihood_in_regime(theory2, regime, num_iter=int(len(states)))

 
last_state = states[-1] #int(np.argmin(dpgmm_model.means_.flatten()) / 2)
count_lowest = lambda St: np.sum([1 if M == last_state else 0 for M in St ])

df=pd.DataFrame()
df['states'] = states
df['state_freq'] = df['states'].rolling(30).apply(count_lowest)

import matplotlib.pyplot as plt
import seaborn as sns

# plot sample2 data
print(sample2)
fig,ax=plt.subplots()
sns.lineplot(data=X['close'].cumsum(), ax=ax, label='baseline', alpha=0.98, color='darkgrey')
sns.lineplot(x=range(len(X), len(X)+len(sample2)), y=np.cumsum([s[0] for s in sample2]), ax=ax, label='theory', alpha=0.48, color='darkred')

# set plot labels and title
ax.set_xlabel(f'base data [{startdate} through {enddate}]')
ax.set_title('SPY Reality + SPY Theory')

# plot configurations
plt.grid(None)
plt.show()

# plot sample_likelihood1 and sample_likelihood2
fig,ax=plt.subplots()
wlen_roll=40
sns.lineplot(data=sample_likelihood1, alpha=0.95, color='darkgrey', ax=ax, label='baseline')
sns.lineplot(data=sample_likelihood2, alpha=0.4, color='red', ax=ax, label='theory')

# set plot labels and title
plt.title('mean likelihood')
ax.set_xlabel('samples over time')
ax.set_ylabel('mean likelihood')

# plot configurations
plt.legend(loc='upper left')
plt.show()

