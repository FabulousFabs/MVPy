"""
1. Classification (RidgeClassifier, SVC)
=================

We will begin our look at decoders by using an MEG dataset of humans performing a visual categorization task. Briefly, participants saw a list of 92 images. Here, we will only use 24 of these images, either of faces or not faces. For more information, consult `MNE's documentation <https://mne.tools/1.8/auto_examples/decoding/decoding_rsa_sgskip.html>`_ or the `original paper <https://dx.doi.org/10.1038/nn.3635>`_.

For convenience, we will consider only gradiometer channels from the dataset, though you are free to change this, of course. The goal will be to build a classifier that can distinguish between faces/not-faces.

First, we will have to load the data---To do this, we use MNE's sample code. Be aware that this will download roughly 6GB of data, which may take a while.
"""
#%%
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt

import mne
from mne.datasets import visual_92_categories
from mne.io import concatenate_raws, read_raw_fif

print(__doc__)

data_path = visual_92_categories.data_path()

# Define stimulus - trigger mapping
fname = data_path / "visual_stimuli.csv"
conds = read_csv(fname)
print(conds.head(5))

max_trigger = 24
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

# %% [markdown]
# 
# Now that we have succesfully loaded the data, we will quickly bring the data into a format that we can work with (i.e., arrays) and produce a vector of target labels as well:

#%%
X_nf = epochs['not-face'].get_data(picks = 'grad') # grab data from not-face epochs
X_if = epochs['face'].get_data(picks = 'grad') # grab data from face epochs

# concatenate the data to make it easy to label
# i.e., data is (trials, channels, time)
X = np.concatenate((X_nf, X_if), axis = 0)

# create labels
y = [0] * 360 + [1] * 360

# for model fitting, we want labels to match the dimensions of our data, i.e. (trials, channels, time)
y = np.array(y)[:,None,None] * np.ones((X.shape[0], 1, X.shape[-1]))

print(X.shape, y.shape)

# %% [markdown]
# 
# Now that we have our data in a nicely structured format, let's look at building our classifier. We will build our classifier using torch. Consequently, we will begin by transforming our data to torch tensors. Note that, if you have a GPU, you may also specify a different device than 'cpu' here.
# Next, we will create a relatively standard pipeline using a combination of MVPy estimators and sklearn utilities.

#%%
import torch
from mvpy.math import accuracy
from mvpy.estimators import Scaler, Covariance, Sliding, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

# transform our data to torch on specified device
device = 'cpu' # if desired, change your device here
X_tr, y_tr = torch.from_numpy(X).float().to(device), torch.from_numpy(y).float().to(device)
alphas = torch.logspace(-5, 10, 20).float().to(device)

# define our cross-validation scheme
n_splits = 5
skf = StratifiedKFold(n_splits = n_splits)

# define our classifier pipeline:
#    1. Apply a standard scaling to our data to ensure that all features are on the same scale
#    2. Compute the covariance matrix of our data and use it to whiten our data.
#    3. Create a sliding estimator that will slide a classifier over the dimension of time for us. We also specify a range of alpha values for our RidgeClassifier to test, of course.
pipeline = make_pipeline(
    Scaler().to_torch(),
    Covariance(
        method = 'LedoitWolf',
        s_max = 100
    ).to_torch(),
    Sliding(
        RidgeClassifier(
            alphas
        ),
        dims = torch.tensor([-1]),
        n_jobs = None,
        verbose = True
    )
)

# setup our output data structures: out-of-sample accuracy, and the patterns the classifier used.
oos = torch.zeros((n_splits, X.shape[-1]), dtype = X_tr.dtype, device = X_tr.device) # (folds, time points)
patterns = torch.zeros((n_splits, X.shape[1], X.shape[-1]), dtype = X_tr.dtype, device = X_tr.device) # (folds, channels, time points)

# loop over cross-validation folds
for f_i, (train, test) in enumerate(skf.split(X_tr[:,0,0], y_tr[:,0,0])):
    # fit model
    pipeline.fit(X_tr[train], y_tr[train])
    
    # score model
    y_h = pipeline.predict(X_tr[test])
    oos[f_i] = accuracy(
        y_h.squeeze().t(), 
        y_tr[test].squeeze().t()
    )
    
    # obtain pattern
    pattern = pipeline[2].collect('pattern_')
    pattern = torch.linalg.inv(pipeline[1].whitener_) @ pattern
    pattern = pipeline[0].inverse_transform(pattern[:,:,0].T[None,:,:])
    patterns[f_i] = pattern.squeeze()

# %% [markdown]
# 
# Once we have fit our model, we may now wish to look at how well it performed. Let's do so now.

#%%
fig, ax = plt.subplots()
t = np.arange(-0.1, 0.5 + 1e-3, 1e-3) # epochs range from -100ms to +500ms
ax.plot(t, [0.5] * oos.shape[-1], color = 'red', label = 'chance') # plot chance-level
ax.plot(t, oos.cpu().numpy().mean(axis = 0), label = 'classifier') # plot classifier performance
ax.set_ylabel(r'Accuracy')
ax.set_xlabel(r'Time (s)')
ax.legend(loc = 'upper left')
fig.tight_layout()

# %% [markdown]
# 
# Finally, we would also like to visualise the patterns that the classifier used. Let's visualise this briefly using MNE:

#%%
parray = mne.EpochsArray(patterns, info = epochs.info).average()
parray.plot_topomap(ch_type = "grad");

# %% [markdown]
# 
# Wonderful. However, ridge classifiers are, of course, only one kind of classifier. While they are often powerful enough (and have the advantage of being very efficient, computationally), we might also want to try and use a more powerful classifier. Therefore, let's repeat this procedure with a support vector machine:

# %%
import torch
from mvpy.math import accuracy
from mvpy.estimators import Scaler, Covariance, Sliding, SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

# transform our data to torch on specified device
device = 'cpu' # if desired, change your device here
X_tr, y_tr = torch.from_numpy(X).float().to(device), torch.from_numpy(y).float().to(device)
alphas = torch.logspace(-5, 10, 20).float().to(device)

# define our cross-validation scheme
n_splits = 5
skf = StratifiedKFold(n_splits = n_splits)

# define our classifier pipeline:
#    1. Apply a standard scaling to our data to ensure that all features are on the same scale
#    2. Compute the covariance matrix of our data and use it to whiten our data.
#    3. Create a sliding estimator that will slide a classifier over the dimension of time for us. This time, we specify a SVC with a radial basis kernel.
pipeline = make_pipeline(
    Scaler().to_torch(),
    Covariance(
        method = 'LedoitWolf',
        s_max = 100
    ).to_torch(),
    Sliding(
        SVC(
            kernel = 'rbf'
        ).to_torch(),
        dims = torch.tensor([-1]),
        n_jobs = None,
        verbose = True
    )
)

# setup our output data structures: out-of-sample accuracy, and the patterns the classifier used.
oos = torch.zeros((n_splits, X.shape[-1]), dtype = X_tr.dtype, device = X_tr.device) # (folds, time points)

# loop over cross-validation folds
for f_i, (train, test) in enumerate(skf.split(X_tr[:,0,0], y_tr[:,0,0])):
    # fit model
    pipeline.fit(X_tr[train], y_tr[train])
    
    # score model
    y_h = pipeline.predict(X_tr[test])
    oos[f_i] = accuracy(
        y_h.squeeze().t(), 
        y_tr[test].squeeze().t()
    )
    
    # unfortunately, for non-linear kernel functions, we cannot estimate coefficients or patterns

# %% [markdown]
# 
# Again, let's look at how the model performed:

# %% 
fig, ax = plt.subplots()
t = np.arange(-0.1, 0.5 + 1e-3, 1e-3) # epochs range from -100ms to +500ms
ax.plot(t, [0.5] * oos.shape[-1], color = 'red', label = 'chance') # plot chance-level
ax.plot(t, oos.cpu().numpy().mean(axis = 0), label = 'SVC (rbf)') # plot classifier performance
ax.set_ylabel(r'Accuracy')
ax.set_xlabel(r'Time (s)')
ax.legend(loc = 'upper left')
fig.tight_layout()

# %% [markdown]
# Okay, does not seem to have made much of a difference, but oh well!