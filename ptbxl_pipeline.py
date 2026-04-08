# =============================================================================
# FILE: ptbxl_pipeline.py
# PROJECT: Trustworthy Arrhythmia Diagnosis
# DESCRIPTION: Complete PTB-XL dataset pipeline for independent replication
#              of HCTG-Net results on a second dataset.
#
#              PTB-XL is the world's largest publicly available ECG dataset
#              with 21,837 clinical 12-lead ECG records from 18,885 patients.
#              We use Lead II only for direct comparability with MIT-BIH.
#
#              Pipeline:
#                1. Download PTB-XL from PhysioNet automatically
#                2. Map PTB-XL diagnostic labels to 5 AAMI classes
#                3. Segment beats using Pan-Tompkins R-peak detection
#                4. Z-score normalise + SMOTE balance
#                5. Train HCTG-Net from scratch on PTB-XL
#                6. Evaluate and compare with MIT-BIH results
#
# USAGE:
#   python ptbxl_pipeline.py
#
# OUTPUT:
#   ./results/ptbxl/
#       ptbxl_confusion_matrix.png
#       ptbxl_roc_curves.png
#       ptbxl_learning_curves.png
#       ptbxl_classification_report.txt
#       ptbxl_vs_mitbih_comparison.png
#       best_ptbxl_model.pth
#
# NOTE: PTB-XL is ~2GB. First run will download automatically.
#       Subsequent runs use local cache.
# =============================================================================

import os
import ast
import time
import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
from streamlit import config
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from model import HCTGNet


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

class PTBXLConfig:
    # --- Data ---
    data_dir : str = './ptbxl_data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    results_dir     : str   = './results/ptbxl'
    sampling_rate   : int   = 100    # PTB-XL has 100Hz and 500Hz versions
                                     # we use 100Hz — closest to MIT-BIH 125Hz
    lead_index      : int   = 1      # Lead II index in PTB-XL (0-indexed)
                                     # PTB-XL lead order: I,II,III,aVR,aVL,aVF,V1-V6

    # --- Segmentation ---
    # At 100Hz: 90 pre + 98 post = 188 samples = 1.88 seconds per beat
    # Same window as MIT-BIH for direct model compatibility
    pre_r_samples   : int   = 90
    post_r_samples  : int   = 98
    segment_len     : int   = 188

    # --- Training ---
    num_classes     : int   = 5
    epochs          : int   = 30
    batch_size      : int   = 256
    num_workers     : int   = 0
    learning_rate   : float = 1e-3
    weight_decay    : float = 1e-4
    grad_clip       : float = 1.0
    lr_factor       : float = 0.5
    lr_patience     : int   = 5
    lr_min          : float = 1e-6

    # --- Model ---
    d_model         : int   = 128
    n_heads         : int   = 4
    ffn_dim         : int   = 256
    n_layers        : int   = 2
    dropout         : float = 0.1
    clf_dropout     : float = 0.3

    # --- Splits ---
    test_size       : float = 0.20
    val_size        : float = 0.20
    seed            : int   = 42

    # --- MIT-BIH reference results (for comparison table) ---
    mitbih_accuracy : float = 0.9874
    mitbih_f1       : float = 0.9256
    mitbih_auc      : float = 0.9869

# AAMI class names
CLASS_NAMES = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}


# =============================================================================
# SECTION 2: PTB-XL LABEL TO AAMI MAPPING
# =============================================================================

# PTB-XL uses SCP codes for diagnostic labels.
# We map the most common rhythm/morphology codes to AAMI classes.
#
# Reference: Wagner et al. (2020) "PTB-XL, a large publicly available
# electrocardiography dataset", Scientific Data.
#
# SCP Code → AAMI Class mapping:
#   NORM, SR     → N (Normal sinus rhythm)
#   AFIB, AFLT   → S (Supraventricular — atrial fibrillation/flutter)
#   SVTAC, PSVT  → S (Paroxysmal SVT)
#   PAC, JTAC    → S (Premature atrial/junctional)
#   PVC, BIGU    → V (Premature ventricular)
#   VTACH        → V (Ventricular tachycardia)
#   FUSION       → F (Fusion beats)
#   PACE, LBBB   → Q (Paced / Bundle branch blocks mapped to Q for simplicity)
#   RBBB         → N (Right BBB mapped to N per AAMI standard)

PTBXL_AAMI_MAPPING = {
    # Class N — Normal
    'NORM' : 0, 'SR'   : 0, 'RBBB' : 0, 'IRBBB': 0,
    'ILBBB': 0, 'CLBBB': 0,

    # Class S — Supraventricular
    'AFIB' : 1, 'AFLT' : 1, 'SVTAC': 1, 'PSVT' : 1,
    'PAC'  : 1, 'JTAC' : 1, 'SVARR': 1, 'SARRH': 1,
    'STACH': 1, 'SBRAD': 1,

    # Class V — Ventricular
    'PVC'  : 2, 'BIGU' : 2, 'TRIGU': 2, 'VTACH': 2,
    'VFIB' : 2, 'VFLUT': 2,

    # Class F — Fusion
    'FUSION': 3,

    # Class Q — Paced / Unknown
    'PACE' : 4, 'LBBB' : 4, 'LAFB' : 4, 'LPFB' : 4,
}


# =============================================================================
# SECTION 3: DEVICE + SEED
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device('cuda')
        print(f"[DEVICE] CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device('mps')
        print("[DEVICE] MPS")
    else:
        d = torch.device('cpu')
        print("[DEVICE] CPU")
    return d


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# SECTION 4: SIMPLE R-PEAK DETECTOR
# =============================================================================

def detect_r_peaks_simple(signal: np.ndarray,
                           fs: int = 100) -> np.ndarray:
    """
    Simple R-peak detection using a threshold-based approach.

    For PTB-XL we use a straightforward peak detection:
    1. Square the signal to emphasise peaks
    2. Apply moving average smoothing
    3. Find peaks above a dynamic threshold
    4. Enforce minimum distance between peaks (refractory period)

    This is simpler than Pan-Tompkins but reliable enough for
    segmenting clean clinical ECG recordings.

    Args:
        signal : 1D ECG signal array
        fs     : Sampling frequency in Hz

    Returns:
        r_peaks: Array of R-peak sample indices
    """
    # Square signal to emphasise QRS
    squared = signal ** 2

    # Moving average window (~150ms)
    window  = int(0.15 * fs)
    kernel  = np.ones(window) / window
    smoothed= np.convolve(squared, kernel, mode='same')

    # Dynamic threshold: 40% of max
    threshold = 0.4 * smoothed.max()

    # Find all samples above threshold
    above = smoothed > threshold

    # Minimum distance between peaks (~300ms refractory period)
    min_dist = int(0.3 * fs)

    r_peaks  = []
    in_peak  = False
    peak_start = 0

    for i in range(len(above)):
        if above[i] and not in_peak:
            in_peak    = True
            peak_start = i
        elif not above[i] and in_peak:
            in_peak = False
            # Find the actual peak within this region
            peak_region = signal[peak_start:i]
            if len(peak_region) > 0:
                local_peak = peak_start + np.argmax(np.abs(peak_region))
                # Enforce minimum distance
                if len(r_peaks) == 0 or (local_peak - r_peaks[-1]) >= min_dist:
                    r_peaks.append(local_peak)

    return np.array(r_peaks)


# =============================================================================
# SECTION 5: PTB-XL DATA LOADING
# =============================================================================

def load_ptbxl_database(config: PTBXLConfig) -> tuple:
    """
    Downloads and loads the PTB-XL database.

    PTB-XL structure:
        - Records stored in subfolders records100/ and records500/
        - Metadata in ptbxl_database.csv
        - SCP (diagnostic) codes in scp_statements.csv

    We use the 100Hz version (records100/) for closest match to MIT-BIH.

    Args:
        config: PTBXLConfig instance

    Returns:
        segments (np.ndarray): Shape (N, 188)
        labels   (np.ndarray): Shape (N,)
    """
    os.makedirs(config.data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Download database metadata if not present
    # ------------------------------------------------------------------
    metadata_path = os.path.join(config.data_dir, 'ptbxl_database.csv')

    # Download metadata files only if missing
    if not os.path.exists(metadata_path):
        print("[DOWNLOAD] Downloading PTB-XL metadata files ...")
        # Only download the metadata CSVs, not all records
        import urllib.request
        base_url = 'https://physionet.org/files/ptb-xl/1.0.3/'
        for fname in ['ptbxl_database.csv', 'scp_statements.csv']:
            fpath = os.path.join(config.data_dir, fname)
            if not os.path.exists(fpath):
                print(f"  Downloading {fname} ...")
                urllib.request.urlretrieve(base_url + fname, fpath)
        print("  Metadata ready!")
    else:
        print("[DATA] PTB-XL metadata found in local cache.")

    # ------------------------------------------------------------------
    # Load metadata CSV
    # ------------------------------------------------------------------
    print("[DATA] Loading PTB-XL metadata ...")
    metadata = pd.read_csv(metadata_path, index_col='ecg_id')

    # Parse SCP codes from string representation of dict
    metadata['scp_codes'] = metadata['scp_codes'].apply(
        lambda x: ast.literal_eval(x)
    )

    print(f"  Total records: {len(metadata):,}")

    # ------------------------------------------------------------------
    # Map SCP codes to AAMI classes
    # ------------------------------------------------------------------
    def get_aami_label(scp_codes: dict) -> int:
        """
        Maps a record's SCP codes to a single AAMI class.
        Uses the highest-confidence SCP code.
        Returns -1 if no mapping found.
        """
        # Sort by confidence (descending)
        sorted_codes = sorted(
            scp_codes.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for code, confidence in sorted_codes:
            if code in PTBXL_AAMI_MAPPING:
                return PTBXL_AAMI_MAPPING[code]
        return -1  # unmapped

    metadata['aami_label'] = metadata['scp_codes'].apply(get_aami_label)

    # Keep only records with valid AAMI mapping
    metadata = metadata[metadata['aami_label'] >= 0]
    print(f"  Records with valid AAMI mapping: {len(metadata):,}")
    print(f"  Class distribution (records): "
          f"{dict(sorted(Counter(metadata['aami_label']).items()))}")

    # ------------------------------------------------------------------
    # Extract beat segments from each record
    # ------------------------------------------------------------------
    all_segments = []
    all_labels   = []
    skipped      = 0
    processed    = 0

    print(f"\n[SEGMENTATION] Extracting beats from "
          f"{len(metadata):,} records ...")
    print("  (This takes ~10-20 minutes on first run)")

    for ecg_id, row in metadata.iterrows():

        # Build record path
        # PTB-XL stores records in subfolders like records100/00000/
        filename = row['filename_lr']  # low-res (100Hz) filename
        record_path = os.path.join(config.data_dir, filename)

        try:
            # Try local cache first
            record = wfdb.rdrecord(record_path)
        except Exception:
            try:
                # Stream from PhysioNet if not cached locally
                record = wfdb.rdrecord(
                    filename.replace('\\', '/'),
                    pn_dir='ptb-xl/1.0.3',
                )
            except Exception:
                skipped += 1
                continue

        # Extract Lead II (index 1 in PTB-XL)
        if record.p_signal is None or record.p_signal.shape[1] <= config.lead_index:
            skipped += 1
            continue

        signal   = record.p_signal[:, config.lead_index].astype(np.float32)
        n_samples= len(signal)
        label    = int(row['aami_label'])

        # Detect R-peaks
        r_peaks = detect_r_peaks_simple(signal, fs=config.sampling_rate)

        if len(r_peaks) == 0:
            skipped += 1
            continue

        # Extract beat segments around each R-peak
        beats_extracted = 0
        for r_peak in r_peaks:
            start = r_peak - config.pre_r_samples
            end   = r_peak + config.post_r_samples

            if start < 0 or end > n_samples:
                continue

            segment = signal[start:end]   # (188,)

            # Basic quality check — reject flat or NaN segments
            if np.isnan(segment).any() or segment.std() < 1e-6:
                continue

            all_segments.append(segment)
            all_labels.append(label)
            beats_extracted += 1

        processed += 1
        if processed % 1000 == 0:
            print(f"  Processed {processed:,}/{len(metadata):,} records  "
                  f"| Beats so far: {len(all_segments):,}")

    segments = np.array(all_segments, dtype=np.float32)
    labels   = np.array(all_labels,   dtype=np.int64)

    print(f"\n[INFO] PTB-XL Extraction complete.")
    print(f"  Total beats extracted : {len(segments):,}")
    print(f"  Records skipped       : {skipped:,}")
    print(f"  Class distribution    : "
          f"{dict(sorted(Counter(labels).items()))}\n")

    return segments, labels


# =============================================================================
# SECTION 6: PREPROCESSING (same as MIT-BIH pipeline)
# =============================================================================

def z_score_normalise(segments: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    """Per-beat Z-score normalisation."""
    mu    = segments.mean(axis=1, keepdims=True)
    sigma = segments.std( axis=1, keepdims=True)
    return ((segments - mu) / (sigma + eps)).astype(np.float32)


def split_dataset(segments, labels, config: PTBXLConfig) -> dict:
    """Stratified train/val/test split."""
    X_tv, X_test, y_tv, y_test = train_test_split(
        segments, labels,
        test_size=config.test_size,
        stratify=labels,
        random_state=config.seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=config.val_size,
        stratify=y_tv,
        random_state=config.seed,
    )
    print("[INFO] PTB-XL split sizes:")
    print(f"  Train : {len(X_train):>7,} | "
          f"{dict(sorted(Counter(y_train).items()))}")
    print(f"  Val   : {len(X_val):>7,} | "
          f"{dict(sorted(Counter(y_val).items()))}")
    print(f"  Test  : {len(X_test):>7,} | "
          f"{dict(sorted(Counter(y_test).items()))}\n")
    return {
        'train': {'X': X_train, 'y': y_train},
        'val'  : {'X': X_val,   'y': y_val  },
        'test' : {'X': X_test,  'y': y_test },
    }


def apply_smote(X_train, y_train, seed=42):
    """SMOTE oversampling on training set only."""
    print("[INFO] Applying SMOTE to PTB-XL training set ...")
    print(f"  Before: {dict(sorted(Counter(y_train).items()))}")
    smote = SMOTE(sampling_strategy='auto', k_neighbors=5,
                  random_state=seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  After : {dict(sorted(Counter(y_res).items()))}\n")
    return X_res, y_res


# =============================================================================
# SECTION 7: PYTORCH DATASET
# =============================================================================

class PTBXLDataset(Dataset):
    """ECG Dataset with optional augmentation for PTB-XL."""

    def __init__(self, segments: np.ndarray,
                 labels: np.ndarray, augment: bool = False):
        self.segments = torch.from_numpy(segments).float()
        self.labels   = torch.from_numpy(labels).long()
        self.augment  = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.segments[idx].clone()

        if self.augment:
            if torch.rand(1) < 0.5:
                signal = signal + torch.randn_like(signal) * 0.02
            if torch.rand(1) < 0.5:
                signal = signal * torch.FloatTensor(1).uniform_(0.9, 1.1)
            if torch.rand(1) < 0.3:
                signal = signal + torch.linspace(-0.05, 0.05, steps=188)
            if torch.rand(1) < 0.3:
                signal = torch.roll(
                    signal, torch.randint(-3, 3, (1,)).item()
                )

        return signal.unsqueeze(0), self.labels[idx]


# =============================================================================
# SECTION 8: TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        running_loss += loss.item() * signals.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_targets.extend(labels.detach().cpu().numpy())

    n = len(loader.dataset)
    return {
        'loss'    : running_loss / n,
        'f1'      : f1_score(all_targets, all_preds,
                             average='macro', zero_division=0),
        'accuracy': accuracy_score(all_targets, all_preds),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        logits  = model(signals)
        loss    = criterion(logits, labels)
        running_loss += loss.item() * signals.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    return {
        'loss'    : running_loss / n,
        'f1'      : f1_score(all_targets, all_preds,
                             average='macro', zero_division=0),
        'accuracy': accuracy_score(all_targets, all_preds),
        'preds'   : np.array(all_preds),
        'targets' : np.array(all_targets),
    }


# =============================================================================
# SECTION 9: RESULT PLOTS
# =============================================================================

def plot_confusion_matrix(targets, preds, save_path):
    cm      = confusion_matrix(targets, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = [CLASS_NAMES[i] for i in range(5)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_title('HCTG-Net on PTB-XL — Normalised Confusion Matrix',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label',      fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


def plot_learning_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('HCTG-Net PTB-XL Training Curves',
                 fontsize=13, fontweight='bold')

    panels = [('Loss', 'train_loss', 'val_loss'),
              ('Accuracy', 'train_acc', 'val_acc'),
              ('Macro F1', 'train_f1', 'val_f1')]

    for ax, (title, tk, vk) in zip(axes, panels):
        ax.plot(epochs, history[tk], 'b-o', markersize=3,
                label='Train', linewidth=1.5)
        ax.plot(epochs, history[vk], 'r-o', markersize=3,
                label='Val',   linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


def plot_dataset_comparison(
    mitbih_acc : float, mitbih_f1  : float, mitbih_auc : float,
    ptbxl_acc  : float, ptbxl_f1   : float, ptbxl_auc  : float,
    save_path  : str,
):
    """Side-by-side bar chart comparing MIT-BIH and PTB-XL results."""
    metrics = ['Accuracy (%)', 'Macro F1 (%)', 'Macro AUC (%)']
    mitbih  = [mitbih_acc*100, mitbih_f1*100, mitbih_auc*100]
    ptbxl   = [ptbxl_acc*100,  ptbxl_f1*100,  ptbxl_auc*100]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars1 = ax.bar(x - width/2, mitbih, width,
                   color='#00b4d8', alpha=0.88, label='MIT-BIH')
    bars2 = ax.bar(x + width/2, ptbxl,  width,
                   color='#00e676', alpha=0.88, label='PTB-XL')

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f'{bar.get_height():.2f}%',
                ha='center', va='bottom',
                color='white', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='white', fontsize=10)
    ax.set_ylabel('Score (%)', color='white', fontsize=11)
    y_min = min(mitbih + ptbxl) - 5
    ax.set_ylim(max(0, y_min), 102)
    ax.set_title(
        'HCTG-Net — MIT-BIH vs PTB-XL Dataset Comparison\n'
        'Independent replication on two datasets',
        color='white', fontsize=12, fontweight='bold', pad=12,
    )
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333355')
    ax.grid(axis='y', linestyle=':', alpha=0.25, color='white')
    ax.legend(fontsize=10, facecolor='#1a1a2e',
              edgecolor='#444466', labelcolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


# =============================================================================
# SECTION 10: MASTER PIPELINE
# =============================================================================

def run_ptbxl_pipeline(config: PTBXLConfig):
    """
    Runs the complete PTB-XL pipeline:
        1. Load + preprocess PTB-XL
        2. Train HCTG-Net from scratch
        3. Evaluate on test set
        4. Compare with MIT-BIH results
        5. Save all plots and reports
    """
    set_seed(config.seed)
    device = get_device()
    os.makedirs(config.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load PTB-XL
    # ------------------------------------------------------------------
    segments, labels = load_ptbxl_database(config)

    # ------------------------------------------------------------------
    # 2. Normalise + Split + SMOTE
    # ------------------------------------------------------------------
    segments = z_score_normalise(segments)
    splits   = split_dataset(segments, labels, config)
    X_tr, y_tr = apply_smote(splits['train']['X'],
                              splits['train']['y'],
                              seed=config.seed)

    # ------------------------------------------------------------------
    # 3. Build DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        PTBXLDataset(X_tr, y_tr, augment=True),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        PTBXLDataset(splits['val']['X'], splits['val']['y']),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        PTBXLDataset(splits['test']['X'], splits['test']['y']),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 4. Model + Loss + Optimiser
    # ------------------------------------------------------------------
    model = HCTGNet(
        num_classes=config.num_classes,
        d_model=config.d_model, n_heads=config.n_heads,
        ffn_dim=config.ffn_dim, n_layers=config.n_layers,
        dropout=config.dropout, clf_dropout=config.clf_dropout,
    ).to(device)

    print(f"[MODEL] HCTG-Net  |  "
          f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    from sklearn.utils.class_weight import compute_class_weight
    # Only compute weights for classes that actually exist in training data
    unique_classes = np.unique(y_tr)
    class_weights_partial = compute_class_weight(
        'balanced', classes=unique_classes, y=y_tr
    )
    # Fill missing classes with weight 1.0
    class_weights = np.ones(config.num_classes)
    for cls, weight in zip(unique_classes, class_weights_partial):
        class_weights[cls] = weight
    print(f"  Classes present in PTB-XL: {unique_classes.tolist()}")
    print(f"  Class weights: { {i: round(float(w),3) for i,w in enumerate(class_weights)} }")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.lr_factor,
        patience=config.lr_patience, min_lr=config.lr_min, verbose=True,
    )

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    best_val_f1  = -1.0
    checkpoint   = os.path.join(config.results_dir, 'best_ptbxl_model.pth')
    history      = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : [],
        'train_f1'  : [], 'val_f1'  : [],
    }

    print(f"[TRAIN] Starting PTB-XL training for {config.epochs} epochs ...")
    print("=" * 72)

    total_start = time.time()

    for epoch in range(1, config.epochs + 1):
        t0      = time.time()
        train_m = train_one_epoch(
            model, train_loader, criterion,
            optimizer, device, config.grad_clip
        )
        val_m   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_m['f1'])
        elapsed = time.time() - t0

        history['train_loss'].append(train_m['loss'])
        history['val_loss'  ].append(val_m['loss'])
        history['train_acc' ].append(train_m['accuracy'])
        history['val_acc'   ].append(val_m['accuracy'])
        history['train_f1'  ].append(train_m['f1'])
        history['val_f1'    ].append(val_m['f1'])

        print(
            f"  Epoch [{epoch:>3}/{config.epochs}]  "
            f"T-Loss: {train_m['loss']:.4f}  T-F1: {train_m['f1']:.4f}  |  "
            f"V-Loss: {val_m['loss']:.4f}  V-F1: {val_m['f1']:.4f}  "
            f"[{elapsed:.0f}s]"
        )

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            torch.save({
                'epoch'           : epoch,
                'model_state_dict': model.state_dict(),
                'val_f1'          : best_val_f1,
            }, checkpoint)
            print(f"    ✓ New best val F1 = {best_val_f1:.4f} — saved")

    total_time = (time.time() - total_start) / 60
    print("=" * 72)
    print(f"[TRAIN] Done. Total: {total_time:.1f} min  |  "
          f"Best val F1: {best_val_f1:.4f}\n")

    # ------------------------------------------------------------------
    # 6. Test evaluation
    # ------------------------------------------------------------------
    print("[TEST] Loading best checkpoint ...")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_m = evaluate(model, test_loader, criterion, device)

    print("=" * 72)
    print("  PTB-XL TEST SET RESULTS")
    print("=" * 72)
    print(f"  Test Loss     : {test_m['loss']:.4f}")
    print(f"  Test Accuracy : {test_m['accuracy']:.4f}")
    print(f"  Test Macro F1 : {test_m['f1']:.4f}")
    print()

   # Use only classes present in test data
    unique_test_classes = sorted(np.unique(test_m['targets']).tolist())
    class_labels = [CLASS_NAMES[i] for i in unique_test_classes]
    report = classification_report(
    test_m['targets'], test_m['preds'],
    labels=unique_test_classes,
    target_names=class_labels, digits=4, zero_division=0,
)
    print(report)

    # Compute AUC
    with torch.no_grad():
        all_probs = []
        for signals, _ in test_loader:
            logits = model(signals.to(device))
            all_probs.extend(
                torch.softmax(logits, dim=1).cpu().numpy()
            )
    all_probs = np.array(all_probs)
    y_bin     = label_binarize(test_m['targets'], classes=list(range(5)))
    try:
        ptbxl_auc = roc_auc_score(y_bin, all_probs,
                                   multi_class='ovr', average='macro')
    except Exception:
        ptbxl_auc = 0.0
    print(f"  Macro AUC     : {ptbxl_auc:.4f}")

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    print("\n[SAVING] Generating plots and reports ...")

    plot_confusion_matrix(
        test_m['targets'], test_m['preds'],
        os.path.join(config.results_dir, 'ptbxl_confusion_matrix.png'),
    )
    plot_learning_curves(
        history,
        os.path.join(config.results_dir, 'ptbxl_learning_curves.png'),
    )
    plot_dataset_comparison(
        mitbih_acc=config.mitbih_accuracy,
        mitbih_f1=config.mitbih_f1,
        mitbih_auc=config.mitbih_auc,
        ptbxl_acc=test_m['accuracy'],
        ptbxl_f1=test_m['f1'],
        ptbxl_auc=ptbxl_auc,
        save_path=os.path.join(
            config.results_dir, 'ptbxl_vs_mitbih_comparison.png'
        ),
    )

    # Save text report
    report_path = os.path.join(
        config.results_dir, 'ptbxl_classification_report.txt'
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("HCTG-Net on PTB-XL Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy : {test_m['accuracy']:.4f}\n")
        f.write(f"Test Macro F1 : {test_m['f1']:.4f}\n")
        f.write(f"Macro AUC     : {ptbxl_auc:.4f}\n\n")
        f.write("Per-Class Report:\n")
        f.write(report)
        f.write("\n\nComparison with MIT-BIH:\n")
        f.write(f"  MIT-BIH Accuracy : {config.mitbih_accuracy:.4f}\n")
        f.write(f"  PTB-XL Accuracy  : {test_m['accuracy']:.4f}\n")
        f.write(f"  MIT-BIH Macro F1 : {config.mitbih_f1:.4f}\n")
        f.write(f"  PTB-XL Macro F1  : {test_m['f1']:.4f}\n")

    print(f"  Report → {report_path}")

    return test_m['accuracy'], test_m['f1'], ptbxl_auc


# =============================================================================
# SECTION 11: ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("  HCTG-Net — PTB-XL Independent Replication Pipeline")
    print("=" * 72)
    print("\n  Dataset  : PTB-XL (21,837 records, Lead II, 100Hz)")
    print("  Model    : HCTG-Net (trained from scratch)")
    print("  Epochs   : 30")
    print("  Goal     : Independent replication for IEEE paper\n")

    cfg = PTBXLConfig()
    acc, f1, auc = run_ptbxl_pipeline(cfg)

    print("\n" + "=" * 72)
    print("  FINAL COMPARISON TABLE (for IEEE paper)")
    print("=" * 72)
    print(f"  {'Dataset':<15} {'Accuracy':>10} {'Macro F1':>10} "
          f"{'Macro AUC':>12}")
    print(f"  {'-'*50}")
    print(f"  {'MIT-BIH':<15} "
          f"{cfg.mitbih_accuracy*100:>9.2f}%"
          f"{cfg.mitbih_f1*100:>10.2f}%"
          f"{cfg.mitbih_auc*100:>11.2f}%")
    print(f"  {'PTB-XL':<15} "
          f"{acc*100:>9.2f}%"
          f"{f1*100:>10.2f}%"
          f"{auc*100:>11.2f}%")
    print(f"  {'-'*50}")
    print(f"\n  Files saved in: {cfg.results_dir}/")
    print("=" * 72)