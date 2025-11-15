import numpy as np
from scipy.signal import welch, find_peaks

bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30)
}

def extract_features(raw, sf=100.0, bands=bands):
    """
    Extract spectral + time-domain features from a preprocessed Raw object.
    """

    data = raw.get_data()
    feats = []

    freqs, psd = welch(data, sf, nperseg=int(sf * 2))

    for fmin, fmax in bands.values():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        bandpower = np.log(np.mean(psd[:, idx], axis=1) + 1e-6)
        feats.append(bandpower)

    feats.append(np.ptp(data, axis=1))
    feats.append(np.mean(np.abs(data), axis=1))
    feats.append(np.array([len(find_peaks(ch)[0]) for ch in data]))

    return np.concatenate(feats)
