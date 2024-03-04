import torch
import gpytorch
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

from russell2000 import fetch_assets, \
    symbol_stdate, \
    symbol_eddate

# Data
sym='NVDA'
assetlist = [ sym ]
num_components = 4
import scipy.stats as stats
rolling_mean = lambda serie, wlen: [ np.mean(serie[x - wlen : x]) for x in range(wlen, len(serie) + 1) ]
X = fetch_assets(assetlist)
print(X)
waveform = stats.zscore(rolling_mean(np.diff(np.log(X.values)), 100))
sns.lineplot(data=np.log(X.values))
plt.show()

inducing_point_count = len(waveform)
# Define a simple 1D dataset
train_x = torch.linspace(0, 1, inducing_point_count)
#train_y = torch.sin(train_x * (2 * 3.1416)) + torch.randn(train_x.size()) * 0.2
train_y = torch.tensor([float(x) for x in waveform])#

# Define the GP regression model
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        kern = gpytorch.kernels.MaternKernel()
        kern.initialize(lengthscale=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(kern)
        
        C = self.covar_module(
            torch.tensor(train_y)).evaluate().detach().numpy() 
        
        # Plot the evaluation results
        _, ax = plt.subplots()
        im = ax.imshow(C, cmap='twilight', origin='lower')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('similarity measure', rotation=-90, va="bottom")
        
        # Set labels
        ax.set_xlabel('time')
        ax.set_ylabel('time')
        ax.set_title('self-similarity matrix')
       
        # Show the plot
        plt.grid(False)
        plt.show()
    
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(train_x)

# Training the model
model.train()
likelihood.train()

# Use the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training iterations
n_iter = 999
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
    optimizer.step()

# Set model and likelihood to eval mode
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, inducing_point_count)
    obs_output = model(test_x)
    obs_pred = likelihood(obs_output)

# Plotting
with torch.no_grad():
    mean = obs_pred.mean
    print(obs_pred.mean)
    lower, upper = obs_pred.confidence_region()

#Yp=obs_output.sample(sample_shape=(1000,))
#print(Yp)
fig,ax=plt.subplots()
ax.scatter(train_x.numpy(), train_y.numpy(), color='k', label='Training Data (mean return)', s=3)
ax2=ax.twinx()
ax.plot(test_x.numpy(), mean.numpy(), label='Mean Prediction')
ax2.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.2, color='blue')
ax2.grid(None)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title(f'{sym} Gaussian Process {symbol_stdate} to {symbol_eddate}')
plt.legend()
plt.show()
