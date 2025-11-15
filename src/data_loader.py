import os
import mne

def load_raw_files(data_dir="data/raw"):
    """
    Load EDF files from data_dir and assign labels based on filename suffix.
    _1.edf = responder (1)
    _2.edf = non-responder (0)
    """

    files = [f for f in os.listdir(data_dir) if f.lower().endswith('.edf')]
    raw_list = []
    labels = []

    for fname in files:
        file_path = os.path.join(data_dir, fname)
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw_list.append(raw)

        if fname.lower().endswith('_1.edf'):
            labels.append(1)
        elif fname.lower().endswith('_2.edf'):
            labels.append(0)
        else:
            raise ValueError(f"Filename {fname} must end with _1 or _2.")

    return raw_list, labels
