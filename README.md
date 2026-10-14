# EEG-Based Antidepressant Treatment Efficacy Prediction

## Overview
This project analyzes EEG signals to predict whether a patient will respond positively to antidepressant treatment (responders vs. non-responders). It was completed as graduate coursework at Northeastern University. The data consists of pre- and post-treatment EEG recordings from a small patient cohort.

The workflow includes:
- Data preprocessing
- Feature extraction
- Neural network modeling
- Model evaluation
- Critical analysis of results

---
## Dataset

This project uses an EEG depression treatment-response dataset (responders vs. non-responders) containing multi-channel EEG recordings of patients before and after antidepressant treatment. The data was provided for the course and is not redistributed in this repository.

To run the pipeline, place the .edf files under: data/raw/

## Project Structure

```
EEG_Signals/
│
├── data/                     # raw and processed EEG files
│
├── src/
│   ├── preprocessing.py      # filtering, cleaning, downsampling
│   ├── feature_extraction.py # bandpower and time-domain feature extraction
│   ├── model.py              # neural network architecture
│   ├── train.py              # training pipeline
│   └── evaluate.py           # evaluation and plotting
│
├── notebooks/   
│
├── README.md
└── requirements.txt
```

## Data Preprocessing
The preprocessing pipeline includes methods such as:
- Notch filtering at 60 Hz
- Band-pass filtering between 1–30 Hz
- Downsampling to 100 Hz
- Artifact rejection
- Channel interpolation
- Mean-value imputation
- Normalization
- Train/validation split

---

## Feature Extraction
Each EEG channel is transformed using both spectral and time-domain features[cite: 1].

### Spectral Features (Welch’s method)
- Delta bandpower (1–4 Hz)
- Theta bandpower (4–8 Hz)
- Alpha bandpower (8–12 Hz)
- Beta bandpower (12–30 Hz)

These values are log-scaled to stabilize variance.

### Time-Domain Features
- Peak-to-peak amplitude
- Mean absolute amplitude
- Peak count (local maxima)

---

## Model Architecture
A feedforward neural network is used:

```python
Dense(64, activation='relu')
Dense(32, activation='relu')
Dense(1, activation='sigmoid')
```
