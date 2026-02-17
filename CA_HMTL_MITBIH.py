
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support



base_path1 = '/mitbih/temporal'

input_filenames = [
    '100.csv','101.csv','102.csv','103.csv','104.csv','105.csv','106.csv','107.csv',
    '108.csv','109.csv','111.csv','112.csv','113.csv','114.csv','115.csv','116.csv',
    '117.csv','118.csv','119.csv','121.csv','122.csv','123.csv','124.csv',
    '200.csv','201.csv','202.csv','203.csv','205.csv','207.csv','208.csv','209.csv',
    '210.csv','212.csv','213.csv','214.csv','215.csv','217.csv','219.csv','220.csv',
    '221.csv','222.csv','223.csv','228.csv','230.csv','231.csv','232.csv','233.csv','234.csv'
]

files = [f.replace('.csv', '_MLII_filtered.csv') for f in input_filenames]

dfs = []
for f in files:
    df = pd.read_csv(os.path.join(base_path1, f), header=None)
    dfs.append(df)   # <<< ONLY FIRST 300 RECORDS

all_features = sorted(set(col for df in dfs for col in df.columns[3:]))

for i in range(len(dfs)):
    dfs[i] = dfs[i].reindex(columns=[0,1,2] + all_features, fill_value=0)

data = pd.concat(dfs, ignore_index=True)

AAMI_MAP = {
    # Normal
    'N':'N','L':'N','R':'N','e':'N','j':'N',
    # Supraventricular
    'A':'S','a':'S','J':'S','S':'S',
    # Ventricular
    'V':'V','E':'V',
    # Fusion
    'F':'F',
    # Unknown / paced / noise
    '/':'Q','f':'Q','Q':'Q','|':'Q'
}

def map_aami_labels(label_series, mapping):
    """
    Map raw ECG beat labels to AAMI classes.

    Parameters:
        label_series (pd.Series): Original labels
        mapping (dict): AAMI mapping dictionary

    Returns:
        pd.Series: Mapped labels (NaN for unmapped)
    """
    return label_series.str.strip().map(mapping)

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Extract raw labels
raw_labels = data.iloc[:, 2]

# Apply AAMI mapping
labels = map_aami_labels(raw_labels, AAMI_MAP)

# Keep only valid AAMI classes
labels_to_keep = ['N', 'S', 'V', 'F', 'Q']
mask = labels.isin(labels_to_keep)

# Filter data
data = data[mask]
labels = labels[mask]

# Extract features
features = data.iloc[:, 3:]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=30)


print(X.shape)
for tr, te in sss.split(X, y):
    X_train, X_test = X[tr], X[te]
    y_train, y_test = y[tr], y[te]



# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Convert NumPy → Torch
# ------------------------
if isinstance(X_train, np.ndarray):
    X_train = torch.from_numpy(X_train)
if isinstance(X_test, np.ndarray):
    X_test = torch.from_numpy(X_test)
if isinstance(y_train, np.ndarray):
    y_train = torch.from_numpy(y_train)

# ------------------------
# Ensure SEQUENCE shape
# ------------------------
if X_train.ndim == 2:
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

X_train = X_train.float().clone().detach()
X_test  = X_test.float().clone().detach()
y_train = y_train.long().clone().detach()

# ------------------------
# Sinusoidal Positional Encoding
# ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# ------------------------
# Transformer Classifier
# ------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        d_model = 128

        self.embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global Average Pooling
        return self.fc(x)

# ------------------------
# DataLoader
# ------------------------
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,
    pin_memory=True
)

# ------------------------
# Model / Loss / Optimizer
# ------------------------
model = TransformerClassifier(
    input_dim=X_train.shape[-1],
    num_classes=len(le.classes_)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

scaler = torch.cuda.amp.GradScaler()

# ------------------------
# Training Loop
# ------------------------
for epoch in range(50):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

# ------------------------
# Inference
# ------------------------
@torch.no_grad()
def transformer_probs(model, X):
    model.eval()
    X = X.to(device)
    logits = model(X)
    return torch.softmax(logits, dim=1).cpu().numpy()

tf_probs = transformer_probs(model, X_test)

# ----------------------------------------
# Load all PQRST feature files
# ----------------------------------------
files = [f.replace('.csv', '_PQRST_with_class.csv') for f in input_filenames]

base_path='/mitbih/morphological'
dfs = []
for f in files:
    df = pd.read_csv(os.path.join(base_path, f))
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)


# Apply mapping on last column (class label)
labels = map_aami_labels(data.iloc[:, -1], AAMI_MAP)

# Keep only standard AAMI classes
labels_to_keep = ['N', 'S', 'V', 'F', 'Q']
mask = labels.isin(labels_to_keep)

# Filter data
data = data[mask]
labels = labels[mask]

print("Data shape after AAMI filtering:", data.shape)

# ----------------------------------------
# Feature scaling & label encoding
# ----------------------------------------
X_rf = StandardScaler().fit_transform(data.iloc[:, 2:-1])
y_rf = le.transform(labels)

for tr, te in sss.split(X_rf, y_rf):
    Xr_tr, Xr_te = X_rf[tr], X_rf[te]
    yr_tr, yr_te = y_rf[tr], y_rf[te]

rf = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=100, random_state=42), # n_estimators=100
    method='sigmoid', cv=5
)
rf.fit(Xr_tr, yr_tr)

rf_probs = rf.predict_proba(Xr_te)

"""#### Voting ensemble"""

############ Hard Voting Ensemble##################


# Individual model predictions
tf_pred = np.argmax(tf_probs, axis=1)
rf_pred = np.argmax(rf_probs, axis=1)

# Stack predictions: shape (n_samples, n_models)
all_preds = np.vstack([tf_pred, rf_pred]).T

# Majority voting
Majority_voting_ensemble_y_pred = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(),
    axis=1,
    arr=all_preds
)

############ soft voting ##################

# Average probabilities
ensemble_probs = (tf_probs + rf_probs) / 2

# Final prediction
Soft_voting_ensemble_y_pred = np.argmax(ensemble_probs, axis=1)

"""Dynamic Confidence Aware Ensemble"""

#============Dynamic weighted ensemble==============================
tf_conf = tf_probs.max(axis=1)
rf_conf = rf_probs.max(axis=1)

print(tf_conf )
print(rf_conf)
w_tf = tf_conf / (tf_conf + rf_conf)
w_rf = rf_conf / (tf_conf + rf_conf)

ensemble_probs = (
    w_tf[:,None] * tf_probs +
    w_rf[:,None] * rf_probs
)

# weighted ensemble
Dynamic_weighted_ensemble_y_pred = np.argmax(ensemble_probs, axis=1)

"""Entropy based ensemble"""

############################# Entropy based ensemble #########################################

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# INPUTS (already computed from your models)
# ------------------------------------------------------------
# tf_probs : (N, C) Transformer softmax probabilities
# rf_probs : (N, C) Random Forest predict_proba outputs
# y_test   : (N,) ground-truth labels
# ============================================================


# ----------------------------
# Utility: Entropy
# ----------------------------
def entropy(p):
    """
    p: (N, C) probability matrix
    returns: (N, 1) entropy
    """
    return -np.sum(p * np.log(p + 1e-8), axis=1, keepdims=True)


# ----------------------------
# Class-Wise Confidence Gating
# ----------------------------
def Entropy_based_ensemble(p_tf, p_rf):
    """
    p_tf, p_rf: (N, C) probability matrices
    returns:
        fused_probs: (N, C)
        alpha_tf   : (N, C)
        alpha_rf   : (N, C)
    """

    # 1) Uncertainty (lower entropy => higher confidence)
    ent_tf = entropy(p_tf)
    ent_rf = entropy(p_rf)

    # 2) Class-wise confidence
    conf_tf = p_tf / (ent_tf + 1e-8)
    conf_rf = p_rf / (ent_rf + 1e-8)

    # 3) Class-wise gating weights
    alpha_tf = conf_tf / (conf_tf + conf_rf + 1e-8)
    alpha_rf = 1.0 - alpha_tf

    # 4) Fuse probabilities
    fused_probs = (
        alpha_tf * p_tf +
        alpha_rf * p_rf
    )

    return fused_probs, alpha_tf, alpha_rf


# ============================================================
# RUN FUSION
# ============================================================
fused_probs, w_tf, w_rf = Entropy_based_ensemble(tf_probs, rf_probs)

# Final prediction
y_pred = fused_probs.argmax(axis=1)
Entropy_based_ensemble_pred=y_pred

# ============================================================
# EVALUATION
# ============================================================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ============================================================
# DEBUG / INSPECTION (IMPORTANT)
# ============================================================
print("\nAverage Transformer weight per class:")
print(w_tf.mean(axis=0))

print("\nAverage Random Forest weight per class:")
print(w_rf.mean(axis=0))



def detailed_classification_report(y_true, y_pred, le):
    """
    Prints per-class metrics, per-class accuracy, support,
    and overall metrics including specificity.
    """

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    total_samples = cm.sum()

    # print("\n========== CLASS LABEL MAPPING ==========")
    # for i, cls in enumerate(le.classes_):
    #     print(f"Class {i} → {cls}")

    print("\n========== PER-CLASS METRICS ==========")


    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total_samples - (tp + fp + fn)

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        specificity = tn / (tn + fp + 1e-9)

        # Per-class accuracy
        class_accuracy = (tp + tn) / total_samples

        # Support
        support = cm[i, :].sum()

        print(
            f"Class {i} ({le.classes_[i]}): "
            f"Acc={class_accuracy:.4f}, "
            f"Prec={precision:.4f}, "
            f"Rec={recall:.4f}, "
            f"F1={f1:.4f}, "
            f"Spec={specificity:.4f}, "
            f"Support={support}"
        )



# Random Forest-only prediction
print("\n RF==================")
y_pred_rf = rf_probs.argmax(axis=1)
y_pred=y_pred_rf
detailed_classification_report(y_test, y_pred, le)

# Transformer-only prediction
print("\n transformer==================")
y_pred_tf = tf_probs.argmax(axis=1)
y_pred=y_pred_tf
detailed_classification_report(y_test, y_pred, le)

print("\n Majority_voting_ensemble ==================")
y_pred=Majority_voting_ensemble_y_pred
detailed_classification_report(y_test, y_pred, le)

print("\n Soft_voting_ensemble ==================")
y_pred=Soft_voting_ensemble_y_pred
detailed_classification_report(y_test, y_pred, le)

print("\n Entropy based ensemble ==================")
y_pred=Entropy_based_ensemble_pred
detailed_classification_report(y_test, y_pred, le)


print("\n Dynamic Confidence Guided Ensemble ==================")
y_pred=Dynamic_weighted_ensemble_y_pred
detailed_classification_report(y_test, y_pred, le)