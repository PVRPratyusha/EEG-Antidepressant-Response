import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_raw_files
from preprocessing import preprocess_raw
from feature_extraction import extract_features
from model import build_model

def build_dataset(data_dir="data/raw"):
    raw_list, labels = load_raw_files(data_dir)
    X = []

    for raw in raw_list:
        clean = preprocess_raw(raw)
        feats = extract_features(clean)
        X.append(feats)

    X = np.array(X)
    y = np.array(labels)
    return X, y

def train_model(data_dir="data/raw"):
    X, y = build_dataset(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        verbose=2
    )

    model.save("model.h5")
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_test.npy", y_test)

    return history

if __name__ == "__main__":
    train_model()
