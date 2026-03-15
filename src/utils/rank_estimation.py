"""Ad-hoc methods to estimate the rank of a noisy matrix."""
import numpy as np

def max_curvature_rank(s, smooth_window=3, logscale=True):
    """
    s: singular values (descending).
    Uses curvature on log(s) vs index; smoothing helps with noise.
    Returns: k (1-based), idx (0-based)
    """
    s = np.sort(np.asarray(s))[::-1]
    if logscale:
        y = np.log(np.maximum(s, 1e-12))
    else:
        y = s

    # simple moving average smoothing
    if smooth_window > 1:
        pad = smooth_window // 2
        y_pad = np.pad(y, (pad, pad), mode='edge')
        kernel = np.ones(smooth_window) / smooth_window
        y = np.convolve(y_pad, kernel, mode='valid')

    # discrete second derivative (curvature proxy)
    # curvature is large (more negative) at elbow on a decreasing spectrum
    d2 = np.zeros_like(y)
    d2[1:-1] = y[:-2] - 2*y[1:-1] + y[2:]
    idx = int(np.argmin(d2[1:-1]) + 1)  # avoid edges
    return idx + 1, idx

def kneedle_elbow(s, decreasing=True):
    """
    s: 1D array of singular values (not squared), unsorted or sorted.
    Returns: k (1-based rank), idx (0-based index).
    """
    s = np.asarray(s)
    s = np.sort(s)[::-1] if decreasing else np.sort(s)
    x = np.linspace(0, 1, len(s))
    y = (s - s.min()) / (s.max() - s.min() + 1e-12)  # normalize to [0,1]
    # Distance from the chord connecting endpoints
    # For decreasing spectra, the "knee" maximizes (y - line).
    line = x  # since endpoints mapped to (0,0) and (1,1)
    diff = y - line if decreasing else line - y
    idx = int(np.argmax(diff))
    return idx + 1, idx