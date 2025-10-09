'''
Functions to generate examples of M-/EEG data with continuous predictors that are drawn from categorical classes.
'''

import numpy as np
import torch
import torch.nn.functional as F
import warnings

from ..preprocessing import LabelBinariser
from ..signal import hann_window
from ..math import kernel_rbf
from .make_meeg_layout import make_meeg_layout

from typing import Union

def make_meeg_categorical(n_trials: int = 120, n_channels: int = 64, t_padding: float = 1.0, t_baseline: float = 0.25, t_duration: float = 1.0, t_length: float = 2.0, fs: int = 200, n_background: int = 50, n_features: int = 1, n_cycles: int = 2, n_classes: int = 2, temperature: float = 1.0, sigma: float = 0.2, snr: float = 0.1, gamma: float = 1.0, phi: float = 0.5, backend: str = 'torch', device: str = 'cpu') -> tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Create an M-EEG dataset based on continuous time-varying stimuli that group together into features and classes.
    
    Parameters
    ----------
    n_trials : int, default=120
        How many trials to simulate?
    n_channels : int, default=64
        How many channels to simulate?
    t_padding : float, default=1.0
        How much padding to use (in seconds)? Padding is cut off before returning, but allows background brain signals to stabilise.
    t_baseline : float, default=0.25
        How much time (in seconds) should pass before stimulus turns on?
    t_duration : float, default=1.0
        How much time (in seconds) should pass before stimulus turns off?
    t_length : float, default=2.0
        How much time (in seconds) should we simulate per trial?
    fs : int, default=200
        Sampling frequency (in Hz).
    n_background : int, default=50
        How many background sources to simulate?
    n_features : int, default=1
        How many features should we simulate (each with their own classes)?
    n_cycles : int, default=2
        How many cycles should each TRF include?
    n_classes : int, default=2
        How many classes should we simulate per feature?
    temperature : float, default=1.0
        When computing the sensitivity of each class to each source feature, what temperature should we use for softmax?
    sigma : float, default=0.2
        When simulating individual trials, how much uncertainty (in unit standard deviations) should there be about any trial's class membership?
    snr : float, default=0.1
        What should the signal to noise ratio be? Here, SNR is defined as :math:`\\frac{P_{f}}{P_{b}}` where :math:`P` is power and :math:`f` and :math:`b` refer to features and background, respectively.
    gamma : float, default=1.0
        What gamma should we use in the radial basis kernel applied over channels when creating channel covariances?
    phi : float, default=0.5
        What phi to use when generating auto-correlated stimulus signals? In the AR1 process, this determines the decay of signals, where :math:`X[t+1] = \phi X[t] + E[t+1]`, but applies only to background signals.
    backend : str, default='torch'
        Which backend to use (numpy or torch)?
    device : str, default='cpu'
        What device to use?
    
    Returns
    -------
    X : Union[np.ndarray, torch.Tensor]
        Simulated class labels of shape (n_trials, n_features, n_timepoints).
    y : Union[np.ndarray, torch.Tensor]
        Simulated neural responses of shape (n_trials, n_channels, n_timepoints).
    
    Examples
    --------
    >>> import torch
    >>> import matplotlib.pyplot as plt
    >>> from mvpy.dataset import make_meeg_layout, make_meeg_colours, make_meeg_categorical
    >>> n_channels = 64
    >>> ch_pos = make_meeg_layout(n_channels)
    >>> ch_col = make_meeg_colours(ch_pos).cpu().numpy()
    >>> X, y = make_meeg_categorical(n_channels = n_channels, n_features = 1, n_classes = 2)
    >>> print(X.shape, y.shape, ch_pos.shape)
    torch.Size([120, 1, 400]) torch.Size([120, 64, 400]) torch.Size([64, 3])
    >>> fig, ax = plt.subplots(1, 3, figsize = (12.0, 4.0))
    >>> t = torch.arange(-0.25, 1.75, 1 / 200)
    >>> for i in range(n_channels):
    >>>     mask_0 = ~X[:,0,0].bool()
    >>>     mask_1 = ~mask_0
    >>>     ax[0].plot(t, y[mask_0,i,:].mean(0), c = ch_col[i])
    >>>     ax[1].plot(t, y[mask_1,i,:].mean(0), c = ch_col[i])
    >>> ax[0].set_ylabel(fr'Amplitude (a.u.)')
    >>> ax[0].set_xlabel(fr'Time ($s$)')
    >>> ax[0].set_title(fr'M-/EEG signal for class$_0$')
    >>> ax[1].set_ylabel(fr'Amplitude (a.u.)')
    >>> ax[1].set_xlabel(fr'Time ($s$)')
    >>> ax[1].set_title(fr'M-/EEG signal for class$_1$')
    >>> ax[2].scatter(ch_pos[:,0], ch_pos[:,1], c = ch_col, marker = 'o', s = 250.0)
    >>> ax[2].axis('off')
    >>> ax[2].set_title(fr'Channel positions')
    >>> fig.tight_layout()
    """
    
    # count features/classes
    n_features_ = n_features
    n_features = n_classes * n_features
    
    # count samples
    n_padding = int(t_padding * fs)
    n_baseline = int(t_baseline * fs)
    n_duration = int(t_duration * fs)
    n_timepoints = int(t_length * fs) + n_padding

    # create foreground and background frequencies
    f_bg = torch.arange(1, n_background + 1, device = device).float()
    f_fg = torch.randint(low = 2, high = 5, size = (n_features,), device = device).float()

    # create foreground and background phase shifts
    p_bg = torch.rand(n_background, device = device)
    p_fg = torch.zeros(n_features, device = device)
    p = torch.cat((p_bg, p_fg))

    # merge and compute amplitudes
    f = torch.cat((f_bg, f_fg))
    a = 1 / f

    # compute channel covariances
    ch_pos = make_meeg_layout(n_channels, device = device)
    ch_cov = kernel_rbf(ch_pos, ch_pos, gamma)

    ch_sen_opts = torch.tensor([0] * (n_channels - 2) + [1, -1], device = device).float()
    ch_sen = torch.stack(
        [
            ch_sen_opts[torch.randperm(n_channels)]
            for i in range(n_background + n_features)
        ],
        dim = 1
    )

    ch_cov = torch.bmm(
        (ch_cov[None,...] * ch_sen.T[:,None,:]),
        ch_cov.expand(ch_sen.shape[1], n_channels, n_channels)
    )

    # compute TRFs per channel
    n_trf = (n_cycles * (1 / f * fs)).clip(max = n_timepoints).long()
    n_pad = (n_trf - n_trf.max()).abs()

    ß = torch.stack(
        [
            F.pad(
                hann_window(n_trf[i], device = device) * a[i] * torch.sin(2 * torch.pi * f[i] * torch.linspace(0, 1, n_trf[i], device = device) + p[None,i] * torch.pi),
                (0, n_pad[i])
            )
            for i in range(n_trf.shape[0])
        ],
        dim = 0
    ).expand(n_channels, n_features + n_background, n_trf.max())

    ß = torch.bmm(
        ß.permute(1, 2, 0), 
        ch_cov,
    ).permute(2, 0, 1)

    # normalise TRFs for SNR
    m_p = ß.abs().sum(-1).sum(0)
    b_p = m_p[:n_background].sum()
    f_p = m_p[n_background:].sum()
    snr_f = snr / (f_p / b_p)
    ß[:,:n_background,:] = ß[:,:n_background,:] / snr_f

    # compute X data from AR1
    E = torch.randn((n_trials, n_background + n_features, n_timepoints), device = device)
    E[:,n_background:,0:n_padding+n_baseline] = 0.0
    E[:,n_background:,n_padding+n_baseline+n_duration:] = 0.0
    X = torch.zeros_like(E)

    for i in range(1, n_timepoints):
        X[...,i] = phi * X[...,i-1] + E[...,i]
    
    # setup sensitivities by class
    S = 0.25 * torch.randn((n_features, n_features), device = device)
    S.fill_diagonal_(1.0)
    S = torch.exp(S / temperature) / torch.exp(S / temperature).sum(0, keepdim = True)
    
    # setup classes
    classes = torch.arange(n_trials, device = device) % n_classes
    classes = torch.stack(
        [
            classes[torch.randperm(n_trials)]
            for i in range(n_features_)
        ],
        dim = 1
    )
    
    # setup encoding
    encoding = LabelBinariser().to_torch().fit_transform(classes).float().to(device)
    encoding = encoding @ S.T
    
    # add some uncertainty to encoding of classes per trial
    encoding = encoding + sigma * torch.randn(*encoding.shape, device = device)
    
    # mix X
    X[:,n_background:,:] = encoding[...,None] * X[:,n_background:,:].sum(0, keepdim = True)

    # scale event amplitudes
    a_bg = X[:,:n_background,n_padding+n_baseline:n_padding+n_baseline+n_duration].abs().sum()
    a_fg = X[:,n_background:,n_padding+n_baseline:n_padding+n_baseline+n_duration].abs().sum()
    X[:,:n_background,:] = X[:,:n_background,:] / (snr / (a_fg / a_bg))
    
    # pad X data
    X_p = F.pad(X, (0, n_pad.max()))

    # convolve in frequency domain
    n_fft = 1 << (X_p.shape[-1] + 1).bit_length()
    F_X = torch.fft.rfft(X_p, n = n_fft, dim = -1)
    F_ß = torch.fft.rfft(ß, n = n_fft, dim = -1)

    y = torch.stack(
        [
            torch.fft.irfft(F_X[:,i,...] * F_ß[:,i,None,...], n = n_fft, dim = -1)[...,:X.shape[-1]]
            for i in range(F_ß.shape[1])
        ],
        dim = 0
    ).sum(0).swapaxes(0, 1)

    # normalise y
    y = y / y.std()

    # cut data
    X = X[:,n_background:,n_padding:]
    y = y[...,n_padding:]
    ß = ß[:,n_background:,:]
    
    # make X our labels
    L = classes[...,None].expand(classes.shape[0], classes.shape[1], y.shape[-1]).float()
    X = L.clone()
    
    # if desired, convert to numpy
    if backend == 'numpy':
        X, y, ß = X.cpu().numpy(), y.cpu().numpy(), ß.cpu().numpy()
    
    return (X, y)