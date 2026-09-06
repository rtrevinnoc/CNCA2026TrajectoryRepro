import argparse
import os
import pickle
import sys

import numpy as np
import pandas.core.indexes
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.cnn_model import WaferCNN

sys.modules['pandas.indexes'] = pandas.core.indexes

LABEL_MAP = {
    'none': 0, 'Center': 1, 'Donut': 2, 'Edge-Loc': 3, 'Edge-Ring': 4,
    'Loc': 5, 'Near-full': 6, 'Random': 7, 'Scratch': 8,
}
LABEL_NAMES = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]


def load_and_preprocess_data(pickle_path):
    with open(pickle_path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')

    def get_failure_type(x):
        if isinstance(x, np.ndarray) and x.size > 0:
            return str(x[0][0])
        if isinstance(x, list) and len(x) > 0:
            return str(x[0][0])
        return None

    df['failureType_str'] = df['failureType'].apply(get_failure_type)
    df = df[df['failureType_str'].notnull()]
    df['label'] = df['failureType_str'].map(LABEL_MAP)
    df = df[df['label'].notnull()]
    df['label'] = df['label'].astype(int)
    return df


def stratified_split(X, y, test_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class WaferDataset(Dataset):
    def __init__(self, wafer_maps, labels):
        self.wafer_maps = wafer_maps
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import cv2
        img = self.wafer_maps[idx]
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
        img = np.expand_dims(img.astype(np.float32), axis=0)
        return torch.from_numpy(img), torch.tensor(self.labels[idx], dtype=torch.long)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="LSWMD.pkl")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common", "wafer_cnn.pth"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    df = load_and_preprocess_data(args.data)
    counts = df['failureType_str'].value_counts()
    print("Dataset composition:")
    for name in LABEL_NAMES:
        print(f"  {name:<10s} {counts.get(name, 0)}")
    print(f"  {'Total':<10s} {len(df)}")

    X = df['waferMap'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, seed=42)

    train_loader = DataLoader(WaferDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(WaferDataset(X_test, y_test),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = WaferCNN(num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / len(train_loader))

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy after epoch {epoch+1}: {accuracy:.2f}%")
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), args.out)
            print(f"Best model saved to {args.out} with accuracy: {best_acc:.2f}%")

    print(f"Final Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
