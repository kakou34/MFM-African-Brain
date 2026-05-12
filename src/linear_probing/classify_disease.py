from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import random


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed()


class EnhancedLinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3, use_batch_norm=True):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x


def _train_model(X_t, y_t, input_dim, lr, epochs, batch_size):
    model = EnhancedLinearClassifier(input_dim=input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for bX, by in loader:
            optimizer.zero_grad()
            criterion(model(bX), by).backward()
            optimizer.step()
    return model


def prepare_features(df, mode):
    """
    Reduce the dataframe to one row per subject according to mode:
      'mean'   - mean-pool feature vectors across all sequences
      'concat' - concatenate feature vectors across sequences
      <seq>    - keep only rows where seq == <seq> (e.g. 't1c')

    Returns (subject_df, feature_cols).
    """
    df = df.copy()
    orig_feats = [c for c in df.columns if c.startswith("Feature")]

    # Derive subject_id and seq from id when not already present.
    # BraTS-style ids are "<subject>_<seq>" (e.g. "BraTS-SSA-00002-000_t1c");
    # single-sequence ids (e.g. "sub-01") contain no underscore.
    if 'subject_id' not in df.columns:
        if df['id'].str.contains('_').any():
            split = df['id'].str.rsplit('_', n=1)
            df['subject_id'] = split.str[0]
            df['seq'] = split.str[1]
        else:
            df['subject_id'] = df['id']

    group_col = 'subject_id'

    if mode == 'mean':
        subject_df = (
            df.groupby(group_col, sort=True)
            .agg(label=('label', 'first'), **{f: (f, 'mean') for f in orig_feats})
            .reset_index()
        )

    elif mode == 'concat':
        if 'seq' not in df.columns:
            raise ValueError("mode='concat' requires multiple sequences but none found in this dataset")
        seqs = sorted(df['seq'].unique())
        labels = df.groupby(group_col, sort=True)['label'].first()
        seq_parts = [
            df[df['seq'] == seq].set_index(group_col)[orig_feats].add_suffix(f'_{seq}')
            for seq in seqs
        ]
        subject_df = (
            pd.concat(seq_parts, axis=1)
            .dropna()               # drop subjects missing any sequence
            .join(labels)
            .reset_index()
        )

    else:
        # Single-sequence mode
        if 'seq' not in df.columns:
            raise ValueError(f"mode='{mode}' requires a 'seq' column but this dataset has only one sequence")
        valid = df['seq'].unique().tolist()
        if mode not in valid:
            raise ValueError(f"mode '{mode}' not in available sequences: {valid}")
        subject_df = df[df['seq'] == mode][[group_col, 'label'] + orig_feats].copy()

    feat_cols = [c for c in subject_df.columns if c.startswith("Feature")]
    return subject_df, feat_cols


def classify(df, mode='mean', n_splits=5, lr=0.0001, epochs=50,
             batch_size=64, out_path="results_disease_classification.csv"):

    subject_df, feat_cols = prepare_features(df, mode)
    print(f"Mode: {mode} | subjects: {len(subject_df)} | features: {len(feat_cols)}")

    X = subject_df[feat_cols].values.astype(np.float32)
    y = subject_df['label'].values.astype(np.float32)

    # Fixed seed → identical subject splits across all model/mode runs.
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=42)
    results = []

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train_np, X_test_np = X[train_idx], X[test_idx]
        y_train_np, y_test_np = y[train_idx], y[test_idx]

        X_train_t = torch.from_numpy(X_train_np).to(DEVICE)
        y_train_t = torch.from_numpy(y_train_np).to(DEVICE).unsqueeze(1)
        X_test_t = torch.from_numpy(X_test_np).to(DEVICE)

        model = _train_model(X_train_t, y_train_t, X_train_np.shape[1], lr, epochs, batch_size)

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

        bal_acc = balanced_accuracy_score(y_test_np, preds)
        f1 = f1_score(y_test_np, preds)
        auc = roc_auc_score(y_test_np, probs)

        print(f"  Split {split_idx + 1}: bal_acc={bal_acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")
        results.append({'split': split_idx + 1, 'mode': mode,
                        'bal_acc': bal_acc, 'f1': f1, 'auc': auc})

    df_results = pd.DataFrame(results)
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(df_results[['bal_acc', 'f1', 'auc']].agg(['mean', 'std']).round(3).to_string())
    return results, df_results


def main(args):
    df = pd.read_csv(args.features_csv)
    classify(df, mode=args.mode, n_splits=args.num_splits,
             lr=args.lr, epochs=args.epochs, out_path=args.output_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Disease classification with repeated stratified splits")
    parser.add_argument('--features_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='results_disease_classification.csv')
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--mode', type=str, default='mean',
                        help="Feature aggregation mode: 'mean', 'concat', or a sequence name (e.g. 't1c')")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()
    main(args)
