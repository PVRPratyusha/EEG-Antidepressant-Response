import numpy as np
import mne

def preprocess_raw(raw, sfreq=100.0):
    """
    Preprocess EDF data:
    - Apply standard 10-20 montage for interpolation
    - Downsample to sfreq
    - Mark bad channels
    - Interpolate bads
    - Impute NaNs
    """

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)

    raw.resample(sfreq, npad='auto')

    data = raw.get_data()
    amplitudes = np.ptp(data, axis=1)
    bad_idx = np.where(amplitudes > 300e-6)[0]

    if bad_idx.size:
        raw.info['bads'] = [raw.ch_names[i] for i in bad_idx]

    raw.interpolate_bads(reset_bads=True)

    data = raw.get_data()
    if np.isnan(data).any():
        means = np.nanmean(data, axis=1, keepdims=True)
        raw._data = np.where(np.isnan(data), means, data)

    return raw
