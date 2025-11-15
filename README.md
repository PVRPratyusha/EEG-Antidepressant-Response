# EEG-Based Antidepressant Treatment Efficacy Prediction

## Overview
This project analyzes EEG signals to predict whether a patient will respond positively to antidepressant treatment. The dataset contains pre- and post-treatment EEG recordings.

The workflow includes:
- Data preprocessing
- Feature extraction
- Neural network modeling
- Model evaluation
- Critical analysis of results

---

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


---

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
Each EEG channel is transformed using both spectral and time-domain features.

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

```
Dense(64, activation='relu')
Dense(32, activation='relu')
Dense(1, activation='sigmoid')
```

Training settings:
- Optimizer: Adam
- Loss: Binary cross-entropy
- Metrics: Accuracy, precision, recall, F1-score

---

## Model Evaluation
Key findings:
- Training and validation loss converge near zero.
- Accuracy approaches 95–100%.
- Confusion matrix summary:
  - Precision: 87.5%
  - Recall: 100%
  - F1-score: 93.3%

---

## Critical Analysis
- Small validation set increases the likelihood of inflated metrics.
- One false positive was observed; this has clinical impact.
- Perfect AUC may imply overfitting.
- Cross-validation with a larger dataset is recommended.
- Threshold stability should be tested for robustness.

---

## Running the Project

Install dependencies:
```
pip install -r requirements.txt
```

Train the model:
```
python src/train.py
```

Evaluate:
## Dataset
Place your dataset (zipped or raw) in:
```
data/
```

This folder is ignored by Git.

---

## Requirements
Dependencies required for this project are listed in `requirements.txt`.

