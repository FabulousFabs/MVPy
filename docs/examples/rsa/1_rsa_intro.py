"""
1. Computing RDMs
=================

Before we turn to real data, we'll briefly familiarise ourselves with the syntax around computing RDMs.
"""
#%%
# Now, let's do some imports:
import torch
import matplotlib.pyplot as plt

from mvpy.estimators import RSA
from mvpy.math import euclidean

#%%
# Let's begin by simulating some data.

trials, channels, timepoints = 240, 60, 100
X = torch.normal(0, 1, (trials, channels, timepoints))

# Because we want to see some structure in our RDMs, let's create some similarity.
X = X * torch.sin(2 * torch.pi * torch.linspace(0, 1, trials))[:,None,None]

#%%
# Given our structured data, we are ready to compute our RDM.
rsa = RSA(estimator = euclidean).fit(X)
rsa.transform(X)
print(rsa.rdm_.shape)

#%%
# As you can see, the RDM is now stacked with all comparisons in the first dimension and only the time domain remaining. This is convenient for modeling and model comparisons, but for now we would like to see the full RDM.
rdm = torch.zeros((trials, trials, timepoints))
rdm[rsa.cx_, rsa.cy_] = rdm[rsa.cy_, rsa.cx_ ] = rsa.rdm_

vmax = torch.abs(rsa.rdm_).max()
fig, ax = plt.subplots()
ax.imshow(rdm[...,0], vmin = -vmax, vmax = vmax, cmap = 'RdBu_r')