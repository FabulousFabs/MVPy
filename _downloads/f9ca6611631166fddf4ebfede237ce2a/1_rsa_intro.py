"""
1. Computing RDMs
=================

As we did for classification, we will use an MEG dataset of humans performing a visual categorization task. Briefly, participants saw a list of 92 images. Effectively, these images are faces, not faces, human, not human, artificial, etc. For more information, consult `MNE's documentation <https://mne.tools/1.8/auto_examples/decoding/decoding_rsa_sgskip.html>`_ or the `original paper <https://dx.doi.org/10.1038/nn.3635>`_. Our goal here will be to create a neural RDM and a hypothesis RDM based on the neural data and the corresponding categories and see if we find some similarity between the two.

First, we will have to load the data---To do this, we use MNE's sample code. Be aware that this will download roughly 6GB of data, which may take a while. As we did in the classification example, we will again be using only gradiometers (for convenience).
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv

import mne
from mne.datasets import visual_92_categories
from mne.io import concatenate_raws, read_raw_fif

print(__doc__)

data_path = visual_92_categories.data_path()

# Define stimulus - trigger mapping
fname = data_path / "visual_stimuli.csv"
conds = read_csv(fname)
print(conds.head(5))

max_trigger = 92
conds = conds[:max_trigger]  # take only the first 24 rows

conditions = []
for c in conds.values:
    cond_tags = list(c[:2])
    cond_tags += [
        ("not-" if i == 0 else "") + conds.columns[k] for k, i in enumerate(c[2:], 2)
    ]
    conditions.append("/".join(map(str, cond_tags)))
print(conditions[:10])

event_id = dict(zip(conditions, conds.trigger + 1))
event_id["0/human bodypart/human/not-face/animal/natural"]

n_runs = 4  # 4 for full data (use less to speed up computations)
fnames = [data_path / f"sample_subject_{b}_tsss_mc.fif" for b in range(n_runs)]
raws = [
    read_raw_fif(fname, verbose="error", on_split_missing="ignore") for fname in fnames
]  # ignore filename warnings
raw = concatenate_raws(raws)

events = mne.find_events(raw, min_duration=0.002)

events = events[events[:, 2] <= max_trigger]

picks = mne.pick_types(raw.info, meg=True)
epochs = mne.Epochs(
    raw,
    events=events,
    event_id=event_id,
    baseline=None,
    picks='grad',
    tmin=-0.1,
    tmax=0.500,
    preload=True,
)

#%%
# Before we can compute RDMs, we would like to also get some data to base our hypothesis RDMs on. Specifically, we will use the epoch categories (human/nonhuman/natural/artificial, bodypart/face/inanimate, human/not-human, face/not-face, natural/not-natural) to build features vectors of our images, like so:

# generate some vectors for conditions based on labels
condition_types = {}

for i, cond in enumerate(conditions):
    items = '/'.join(cond.split(' ')).split('/')[1:]
    
    for j, item in enumerate(items):
        if j not in condition_types:
            condition_types[j] = []
        
        if item not in condition_types[j]:
            condition_types[j].append(item)

N = np.array([len(condition_types[i]) for i in range(len(condition_types))]).sum()
condition_vecs = np.zeros((len(conditions), N))

for i, cond in enumerate(conditions):
    items = '/'.join(cond.split(' ')).split('/')[1:]
    
    indx_i = 0
    for j, item in enumerate(items):
        indx_j = np.where(np.array(condition_types[j]) == item)[0]
        condition_vecs[i,indx_i+indx_j[0]] = 1
        indx_i += len(condition_types[j])

# generate vectors for trials based on condition labels
epochs, indc = epochs.equalize_event_counts()

X = epochs.get_data()
y = np.zeros((X.shape[0], N, X.shape[-1]))
L = np.zeros((X.shape[0],))

for i, event in enumerate(epochs.events):
    y[i,:,:] = condition_vecs[event[2] - 1,:,None]
    L[i] = event[2] - 1

# group by condition
unq, counts = np.unique(L, return_counts = True)
X_g = np.zeros((counts[0], len(unq), X.shape[1], X.shape[2]))
y_g = np.zeros((counts[0], len(unq), y.shape[1], y.shape[2]))

for i, unq_i in enumerate(unq):
    indc = np.where(L == unq_i)[0]
    X_g[:,i] = X[indc]
    y_g[:,i] = y[indc]

#%%
# Now that we have our data, let's move everything to torch. Note that, by default, this will try to look for a GPU.

import torch

# convert data
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
X_g, y_g = torch.from_numpy(X_g).to(torch.float32).to(device), torch.from_numpy(y_g).to(torch.float32).to(device)

#%%
# With the setup out of the way, let's create a quick pipeline to compute our neural and hypothesis RDMs:

from mvpy.estimators import Scaler, RSA
from mvpy.math import *
from sklearn.pipeline import make_pipeline

# compute the neural RDM with the following pipeline:
#   1. Scale the data (zero mean and unit variance)
#   2. Compute the neural RDM using the Pearson correlation as our similarity measure
n_rsa = make_pipeline(Scaler().to_torch(),
                      RSA(estimator = pearsonr, 
                          verbose = True, 
                          n_jobs = None).to_torch())
n_rsa.fit(X_g.mean(0))
n_rsa.transform(X_g.mean(0))

# compute the hypothesis RDM; pipeline is same as neural RDM
h_rsa = make_pipeline(Scaler().to_torch(),
                         RSA(estimator = pearsonr, 
                             verbose = True, 
                             n_jobs = None).to_torch())
h_rsa.fit(y_g.mean(0))
h_rsa.transform(y_g.mean(0))

# Now, let's compare the two RDMs we have so far by computing their spearman correlations
rho_pr = spearmanr(n_rsa[1].rdm_.T, h_rsa[1].rdm_.T)

#%%
# With our result in hand, we may find ourselves wanting to look at the results and some of the raw RSMs we computed. Let's do that now:

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (9.0, 4.0))
max_tp = rho_pr.argmax().cpu().numpy()

t = np.arange(-0.1, 0.5+1e-3, 1e-3)     # time points
ax[0].plot(t, 0*t, color = 'red')       # plot chance level
ax[0].plot(t, rho_pr.cpu().numpy())     # plot spearman correlation
ax[0].set_xlabel(r'Time ($s$)')
ax[0].set_ylabel(r'Spearman $\rho$')

ax[1].imshow(n_rsa[1].full_rdm().cpu().numpy()[:,:,max_tp], vmin = -1, vmax = 1, cmap = 'RdBu_r')   # obtain and plot the full neural RDM from our class
ax[1].set_title(fr'nRDM at $t={np.round(max_tp*1e-3 - 0.1, 3)}s$')
ax[1].set_ylabel('Categories')
ax[1].set_xlabel('Categories')

ax[2].imshow(h_rsa[1].full_rdm().cpu().numpy()[:,:,max_tp], vmin = -1, vmax = 1, cmap = 'RdBu_r')   # obtain and plot the full hypothesis RDM from our class
ax[2].set_title(fr'hRDM at $t={np.round(max_tp*1e-3 - 0.1, 3)}s$')
ax[2].set_ylabel('Categories')
ax[2].set_xlabel('Categories')

plt.tight_layout()