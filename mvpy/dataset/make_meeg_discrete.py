'''
Functions to generate examples of M-/EEG data with continuous predictors.
'''

import numpy as np
import torch
import torch.nn.functional as F
import warnings

from ..signal import hann_window
from ..math import kernel_rbf
from .make_meeg_layout import make_meeg_layout

from typing import Union

def make_meeg_discrete(n_trials: int = 120, n_channels: int = 64, t_padding: float = 1.0, t_baseline: float = 0.25, t_duration: float = 1.0, t_length: float = 2.0, fs: int = 200, n_background: int = 50, n_features: int = 3, n_cycles: int = 2, snr: float = 0.1, gamma: float = 1.0, phi: float = 0.5, poisson: bool = True, variable: bool = True, lambda_min: float = 1e-3, lambda_max: float = 0.1, return_Xyß: bool = False, backend: str = 'torch', device: str = 'cpu') -> tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Create an M-EEG dataset based on stimuli defined as discrete events.
    
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
    n_features : int, default=3
        How many feature sources to simulate?
    n_cycles : int, default=2
        How many cycles should each TRF include?
    snr : float, default=0.1
        What should the signal to noise ratio be? Here, SNR is defined as :math:`\\frac{P_{f}}{P_{b}}` where :math:`P` is power and :math:`f` and :math:`b` refer to features and background, respectively.
    gamma : float, default=1.0
        What gamma should we use in the radial basis kernel applied over channels when creating channel covariances?
    phi : float, default=0.5
        What phi to use when generating auto-correlated stimulus signals? In the AR1 process, this determines the decay of signals, where :math:`X[t+1] = \phi X[t] + E[t+1]` and applies only to background features here.
    poisson : bool, default=True
        Should we sample impulses from a Poisson distribution? If False, only one impulse is placed at stimulus onset.
    variable : bool, default=True
        Should each feature be sampled from a different Poisson distribution, allowing for variability in event rates? Only used if Poisson sampling is enabled. 
    lambda_min : float, default=1e-3
        Minimum rate for sampling from Poisson distribution.
    lambda_max : float, default=0.1
        Maximum rate for sampling from Poisson distribution.
    return_Xyß : bool, default=False
        Should we return X, y and ß?
    backend : str, default='torch'
        Which backend to use (numpy or torch)?
    device : str, default='cpu'
        What device to use?
    
    Returns
    -------
    X : Union[np.ndarray, torch.Tensor]
        Simulated signals of shape (n_trials, n_features, n_timepoints).
    y : Union[np.ndarray, torch.Tensor]
        Simulated neural responses of shape (n_trials, n_channels, n_timepoints).
    ß : Optional[Union[np.ndarray, torch.Tensor]]
        Simulated temporal response functions of shape (n_channels, n_features, n_trf).
    
    Examples
    --------
    >>> import torch
    >>> import matplotlib.pyplot as plt
    >>> from mvpy.dataset import make_meeg_layout, make_meeg_colours, make_meeg_discrete
    >>> n_channels = 64
    >>> ch_pos = make_meeg_layout(n_channels)
    >>> ch_col = make_meeg_colours(ch_pos)
    >>> X, y = make_meeg_discrete(n_channels = n_channels)
    >>> print(X.shape, y.shape, ch_pos.shape)
    torch.Size([120, 3, 400]) torch.Size([120, 64, 400]) torch.Size([64, 3])
    >>> fig, ax = plt.subplots(1, 2, figsize = (8.0, 4.0))
    >>> t = torch.arange(-0.25, 1.75, 1 / 200)
    >>> for i in range(n_channels):
    >>>     ax[0].plot(t, y[:,i,:].mean(0), c = ch_col[i])
    >>> ax[0].set_ylabel(fr'Amplitude (a.u.)')
    >>> ax[0].set_xlabel(fr'Time ($s$)')
    >>> ax[0].set_title(fr'M-/EEG signal')
    >>> ax[1].scatter(ch_pos[:,0], ch_pos[:,1], c = ch_col, marker = 'o', s = 250.0)
    >>> ax[1].axis('off')
    >>> ax[1].set_title(fr'Channel positions')
    >>> fig.tight_layout()
    """
    
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

    # set X background from AR1
    E = torch.randn((n_trials, n_background + n_features, n_timepoints), device = device)
    E[:,n_background:,:] = 0.0
    X = torch.zeros_like(E)
    
    for i in range(1, n_timepoints):
        X[...,i] = phi * X[...,i-1] + E[...,i]
    
    # we always want to tag the stimulus onset
    X[:,n_background:,n_padding+n_baseline] = 1.0
    
    if poisson:
        # we may also want poisson events
        if not variable:
            # draw single rate
            r = lambda_max * torch.rand(1, device = device).expand(n_features)
        else:
            # draw variable rates
            r = lambda_max * torch.rand(n_features, device = device)
        
        # make r bound [lambda_min, lambda_max]
        r = r.clip(min = lambda_min)
        
        # sample events
        n_events = (n_duration * r).clip(min = 1.0).long()
        K = int(n_events.max().item())
        draws = torch.rand(n_trials, n_features, n_duration, device = device)
        events = draws.topk(K, dim = 2).indices
        
        # mask for variable k
        mask = (
            torch.arange(K, device = device).view(1, 1, K) < n_events.view(1, -1, 1)
        ).expand(n_trials, -1, -1)
        
        # grab amplitudes
        amp = torch.randn(n_trials, n_features, K, device = device)
        amp = amp * mask
        
        # scatter add events into a temporary tensor
        tmp = torch.zeros(n_trials, n_features, n_duration, device = device)
        tmp.scatter_add_(dim = 2, index = events, src = amp)
                
        # place in X
        X[:,n_background:,n_padding+n_baseline:n_padding+n_baseline+n_duration] += tmp
    
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
    
    # if desired, convert to numpy
    if backend == 'numpy':
        X, y, ß = X.cpu().numpy(), y.cpu().numpy(), ß.cpu().numpy()
    
    # if desired, return with ß
    if return_Xyß:
        return (X, y, ß)
    
    return (X, y)