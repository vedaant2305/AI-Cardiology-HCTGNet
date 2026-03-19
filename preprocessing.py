# =============================================================================
# FILE: preprocessing.py
# PROJECT: Trustworthy Arrhythmia Diagnosis
# DESCRIPTION: Complete data preprocessing pipeline for the MIT-BIH Arrhythmia
#              Database. Handles signal loading, beat segmentation, AAMI class
#              mapping, Z-score normalization, SMOTE balancing (train only),
#              and PyTorch Dataset / DataLoader construction.
#
# DEPENDENCIES:
#   pip install wfdb numpy scikit-learn imbalanced-learn torch
#
# USAGE:
#   Place this file in your project root and run:
#       python preprocessing.py
#   The MIT-BIH data will be auto-downloaded via wfdb on first run.
# =============================================================================

import os
import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter


# =============================================================================
# SECTION 1: GLOBAL CONSTANTS
# =============================================================================

# --- MIT-BIH Record IDs ---
# All 48 record identifiers in the MIT-BIH Arrhythmia Database.
# Records 102, 104, 107, 217 contain paced beats (|) which are non-standard
# but are retained here; they will map to class Q (unclassified) in AAMI.
MIT_BIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124', '200',
    '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234',
]

# --- Beat Segmentation Window ---
# Each heartbeat is centred on its annotated R-peak.
# 90 samples before R-peak + 1 (R-peak itself) + 97 samples after = 188 total.
# Paper specifies: 90 pre-R and 98 post-R = 188 samples total.
PRE_R_SAMPLES  = 90   # samples to the left  of the R-peak
POST_R_SAMPLES = 98   # samples to the right of the R-peak (includes R itself)
SEGMENT_LEN    = PRE_R_SAMPLES + POST_R_SAMPLES  # == 188

# --- AAMI Standard: MIT-BIH Annotation → 5-Class Mapping ---
# Reference: Kachuee et al. (2018), AAMI EC57 standard.
#
# Class N – Normal / Bundle Branch Block / Escape beats
# Class S – Supraventricular ectopic beats
# Class V – Ventricular ectopic beats
# Class F – Fusion beats
# Class Q – Unknown / Paced / Unclassifiable
AAMI_MAPPING = {
    # ---- Class N (label 0) ----
    'N' : 0,  # Normal beat
    'L' : 0,  # Left bundle branch block
    'R' : 0,  # Right bundle branch block
    'e' : 0,  # Atrial escape beat
    'j' : 0,  # Nodal (junctional) escape beat

    # ---- Class S (label 1) ----
    'A' : 1,  # Atrial premature beat
    'a' : 1,  # Aberrant atrial premature beat
    'J' : 1,  # Nodal (junctional) premature beat
    'S' : 1,  # Supraventricular premature beat

    # ---- Class V (label 2) ----
    'V' : 2,  # Premature ventricular contraction
    'E' : 2,  # Ventricular escape beat

    # ---- Class F (label 3) ----
    'F' : 3,  # Fusion of ventricular and normal beat

    # ---- Class Q (label 4) ----
    '/' : 4,  # Paced beat
    'f' : 4,  # Fusion of paced and normal beat
    'Q' : 4,  # Unclassifiable beat
}

# Human-readable class names (used for logging / display)
CLASS_NAMES = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

# --- DataLoader Hyper-parameters ---
BATCH_SIZE  = 256
NUM_WORKERS = 4    # set to 0 if multiprocessing causes issues on Windows
RANDOM_SEED = 42


# =============================================================================
# SECTION 2: DATA LOADING & BEAT SEGMENTATION
# =============================================================================

import time

def download_and_load_mitbih(data_dir: str = './mitbih_data') -> tuple:
    """
    Loads all MIT-BIH records with automatic retry logic to handle
    transient PhysioNet connection drops (common in Colab environments).
    """
    os.makedirs(data_dir, exist_ok=True)

    all_segments  = []
    all_labels    = []
    skipped_beats = 0

    MAX_RETRIES = 5          # attempts per record before giving up
    RETRY_DELAY = 3          # seconds to wait between retries

    print(f"[INFO] Loading MIT-BIH records into '{data_dir}' ...")
    print(f"[INFO] Processing {len(MIT_BIH_RECORDS)} records ...\n")

    for record_id in MIT_BIH_RECORDS:

        record_path = os.path.join(data_dir, record_id)

        # ------------------------------------------------------------------
        # Load with retry loop — handles ChunkedEncodingError / IncompleteRead
        # that are common on Colab due to PhysioNet rate limits or
        # transient TCP drops during streaming.
        # ------------------------------------------------------------------
        record     = None
        annotation = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # --- Try local cache first ---
                if os.path.exists(record_path + '.hea'):
                    record     = wfdb.rdrecord(record_path)
                    annotation = wfdb.rdann(record_path, 'atr')
                else:
                    # --- Stream from PhysioNet ---
                    print(f"  Fetching record {record_id} "
                          f"(attempt {attempt}/{MAX_RETRIES}) ...")
                    record     = wfdb.rdrecord(record_id, pn_dir='mitdb')
                    annotation = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')

                    # --- Cache to disk so future runs skip the download ---
                    wfdb.wrsamp(
                        record_name=record_id,
                        fs=record.fs,
                        units=record.units,
                        p_signal=record.p_signal,
                        fmt=record.fmt,
                        sig_name=record.sig_name,
                        write_dir=data_dir,
                    )
                    wfdb.wrann(
                        record_name=record_id,
                        extension='atr',
                        sample=annotation.sample,
                        symbol=annotation.symbol,
                        write_dir=data_dir,
                    )

                break   # success — exit the retry loop

            except Exception as e:
                print(f"    ⚠ Attempt {attempt} failed for record "
                      f"{record_id}: {type(e).__name__}")
                if attempt < MAX_RETRIES:
                    print(f"    Retrying in {RETRY_DELAY}s ...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"    ✗ Skipping record {record_id} after "
                          f"{MAX_RETRIES} failed attempts.")

        # Skip this record entirely if all retries failed
        if record is None or annotation is None:
            continue

        # --- Extract Lead II signal ---
        signal    = record.p_signal[:, 0].astype(np.float32)
        n_samples = len(signal)

        for r_peak_idx, beat_symbol in zip(annotation.sample, annotation.symbol):

            if beat_symbol not in AAMI_MAPPING:
                skipped_beats += 1
                continue

            start = r_peak_idx - PRE_R_SAMPLES
            end   = r_peak_idx + POST_R_SAMPLES

            if start < 0 or end > n_samples:
                skipped_beats += 1
                continue

            segment = signal[start:end]
            label   = AAMI_MAPPING[beat_symbol]

            all_segments.append(segment)
            all_labels.append(label)

    segments = np.array(all_segments, dtype=np.float32)
    labels   = np.array(all_labels,   dtype=np.int64)

    print(f"\n[INFO] Extraction complete.")
    print(f"  Total beats extracted : {len(segments)}")
    print(f"  Beats skipped         : {skipped_beats}")
    print(f"  Class distribution    : {dict(sorted(Counter(labels).items()))}\n")

    return segments, labels


# =============================================================================
# SECTION 3: Z-SCORE NORMALISATION
# =============================================================================

def z_score_normalise(segments: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Applies per-beat Z-score normalisation as defined in the paper (Equation 1):

        x_norm = (x - μ) / (σ + ε)

    where μ and σ are computed independently for each heartbeat segment.
    This removes inter-subject amplitude variability and stabilises gradients.

    Args:
        segments (np.ndarray): Shape (N, 188) – raw heartbeat segments.
        eps      (float)     : Small constant to prevent division by zero.

    Returns:
        np.ndarray: Shape (N, 188) – normalised heartbeat segments.
    """
    # Compute mean and std along the time axis (axis=1), keeping dims for
    # broadcasting so each of the 188 time-steps is normalised per beat.
    mu    = segments.mean(axis=1, keepdims=True)  # (N, 1)
    sigma = segments.std( axis=1, keepdims=True)  # (N, 1)

    normalised = (segments - mu) / (sigma + eps)
    return normalised.astype(np.float32)


# =============================================================================
# SECTION 4: TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

def split_dataset(
    segments : np.ndarray,
    labels   : np.ndarray,
    test_size: float = 0.20,
    val_size : float = 0.20,
    seed     : int   = RANDOM_SEED
) -> dict:
    """
    Splits the dataset into train, validation, and test subsets using
    stratified sampling to maintain class proportions in every split.

    Split fractions (matching the paper's 80/20 protocol):
        - Test  : 20% of the full dataset
        - Val   : 20% of the remaining 80%  → ~16% of full dataset
        - Train : remainder                 → ~64% of full dataset

    Args:
        segments  : Shape (N, 188).
        labels    : Shape (N,).
        test_size : Fraction of the full dataset reserved for testing.
        val_size  : Fraction of the train+val pool reserved for validation.
        seed      : Random seed for reproducibility.

    Returns:
        dict with keys 'train', 'val', 'test', each mapping to
        {'X': np.ndarray (M, 188), 'y': np.ndarray (M,)}.
    """
    # --- Step 1: Carve out the held-out test set first ---
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        segments, labels,
        test_size=test_size,
        stratify=labels,          # preserve class ratios
        random_state=seed
    )

    # --- Step 2: Split remaining data into train and validation ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,       # val_size fraction of the trainval pool
        stratify=y_trainval,
        random_state=seed
    )

    print("[INFO] Dataset split sizes:")
    print(f"  Train : {len(X_train):>7,} samples  | "
          f"Classes: {dict(sorted(Counter(y_train).items()))}")
    print(f"  Val   : {len(X_val):>7,} samples  | "
          f"Classes: {dict(sorted(Counter(y_val).items()))}")
    print(f"  Test  : {len(X_test):>7,} samples  | "
          f"Classes: {dict(sorted(Counter(y_test).items()))}\n")

    return {
        'train': {'X': X_train, 'y': y_train},
        'val'  : {'X': X_val,   'y': y_val  },
        'test' : {'X': X_test,  'y': y_test },
    }


# =============================================================================
# SECTION 5: SMOTE CLASS BALANCING (TRAINING SET ONLY)
# =============================================================================

def apply_smote(X_train, y_train, seed=42):
    print("\n[INFO] Applying SMOTE to training set ...")
    print(f"  Before SMOTE: {dict(sorted(Counter(y_train).items()))}")

    # Removed n_jobs to fix the TypeError
    smote = SMOTE(
        sampling_strategy='auto',
        k_neighbors=5,
        random_state=seed
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE:  {dict(sorted(Counter(y_resampled).items()))}")
    return X_resampled, y_resampled


# =============================================================================
# SECTION 6: PYTORCH DATASET
# =============================================================================

class ECGDataset(Dataset):
    """
    Custom PyTorch Dataset for ECG heartbeat segments.

    Each sample is a single-channel 1-D heartbeat segment of length 188.
    The channel dimension is added here so the tensor shape matches what
    PyTorch's Conv1d expects: (Batch, Channels, Length) → (B, 1, 188).

    Args:
        segments (np.ndarray): Shape (N, 188) – normalised heartbeat segments.
        labels   (np.ndarray): Shape (N,)     – integer AAMI class labels.
    """

    def __init__(self, segments: np.ndarray, labels: np.ndarray):
        # Store as float32 tensors; Conv1d requires float input.
        # np.float32 → torch.float32 (no extra cast needed)
        self.segments = torch.from_numpy(segments).float()  # (N, 188)
        self.labels   = torch.from_numpy(labels).long()     # (N,)

    def __len__(self) -> int:
        """Returns the total number of heartbeat samples in this split."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns a single (signal, label) pair.

        The signal is unsqueezed along dim=0 to insert the channel dimension:
            (188,) → (1, 188)
        This matches the expected input format of PyTorch's nn.Conv1d:
            input: (Batch, in_channels, L)
        """
        signal = self.segments[idx].unsqueeze(0)  # (1, 188)
        label  = self.labels[idx]                 # scalar tensor
        return signal, label


# =============================================================================
# SECTION 7: DATALOADER FACTORY
# =============================================================================

def build_dataloaders(
    splits     : dict,
    batch_size : int  = BATCH_SIZE,
    num_workers: int  = NUM_WORKERS,
    seed       : int  = RANDOM_SEED
) -> dict:
    """
    Constructs PyTorch DataLoader objects for train, val, and test splits.

    Key design choices:
        - Train loader : shuffle=True  → randomises batch composition each epoch
        - Val loader   : shuffle=False → deterministic evaluation
        - Test loader  : shuffle=False → deterministic, reproducible metrics
        - pin_memory   : True          → speeds up CPU→GPU transfer via page-locked mem
        - drop_last    : True (train)  → avoids unstable BatchNorm on tiny final batch

    Args:
        splits      : dict returned by split_dataset(), containing resampled
                      training data (after SMOTE) and untouched val/test data.
        batch_size  : Number of samples per mini-batch.
        num_workers : Parallel data-loading workers (set 0 for Windows debug).
        seed        : Worker seed for reproducibility.

    Returns:
        dict with keys 'train', 'val', 'test' → DataLoader objects.
    """

    def seed_worker(worker_id):
        """Ensures each DataLoader worker has a deterministic but unique seed."""
        np.random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)

    # --- Construct Dataset objects ---
    train_dataset = ECGDataset(splits['train']['X'], splits['train']['y'])
    val_dataset   = ECGDataset(splits['val']['X'],   splits['val']['y'])
    test_dataset  = ECGDataset(splits['test']['X'],  splits['test']['y'])

    # --- Construct DataLoader objects ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,             # shuffle every epoch for training
        num_workers=num_workers,
        pin_memory=True,          # faster GPU transfer
        drop_last=True,           # drop incomplete final batch (protects BatchNorm)
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,            # no shuffling needed for evaluation
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("[INFO] DataLoaders constructed:")
    print(f"  Train batches : {len(train_loader)}  "
          f"(~{len(train_dataset):,} samples, batch={batch_size}, drop_last=True)")
    print(f"  Val batches   : {len(val_loader)}  "
          f"(~{len(val_dataset):,} samples)")
    print(f"  Test batches  : {len(test_loader)}  "
          f"(~{len(test_dataset):,} samples)\n")

    return {
        'train': train_loader,
        'val'  : val_loader,
        'test' : test_loader,
    }


# =============================================================================
# SECTION 8: MASTER PIPELINE FUNCTION
# =============================================================================

def build_mitbih_pipeline(
    data_dir   : str  = './mitbih_data',
    batch_size : int  = BATCH_SIZE,
    num_workers: int  = NUM_WORKERS,
    seed       : int  = RANDOM_SEED
) -> dict:
    """
    End-to-end pipeline: download → segment → normalise → split → SMOTE →
    Dataset → DataLoader.

    This is the single public entry-point that the training script will call.

    Args:
        data_dir    : Directory for caching MIT-BIH wfdb files.
        batch_size  : Mini-batch size for all DataLoaders.
        num_workers : Parallel workers for data loading.
        seed        : Global random seed.

    Returns:
        dict with keys 'train', 'val', 'test' → DataLoader objects.
    """

    # --- Step 1: Load raw beats from disk (or download first) ---
    segments, labels = download_and_load_mitbih(data_dir)

    # --- Step 2: Z-score normalise ALL segments before splitting ---
    # Normalisation uses per-beat statistics only (no cross-sample leakage),
    # so it is safe to apply before the train/val/test split.
    segments = z_score_normalise(segments)

    # --- Step 3: Stratified train / val / test split ---
    splits = split_dataset(segments, labels, seed=seed)

    # --- Step 4: SMOTE on TRAINING split only ---
    # Validation and test splits remain completely untouched (real data only).
    X_train_sm, y_train_sm = apply_smote(
        splits['train']['X'],
        splits['train']['y'],
        seed=seed
    )
    # Replace training data in splits dict with the SMOTE-resampled version
    splits['train']['X'] = X_train_sm
    splits['train']['y'] = y_train_sm

    # --- Step 5: Build DataLoaders ---
    loaders = build_dataloaders(splits, batch_size, num_workers, seed)

    return loaders


# =============================================================================
# SECTION 9: STANDALONE TEST / SMOKE-TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    """
    Smoke test: runs the complete pipeline and verifies that a batch drawn
    from each DataLoader has the expected shape and dtype.

    Expected output for the training batch:
        Signal tensor shape : torch.Size([256, 1, 188])
        Label tensor shape  : torch.Size([256])
        Signal dtype        : torch.float32
        Label dtype         : torch.int64
    """

    print("=" * 65)
    print("  MIT-BIH Arrhythmia Preprocessing Pipeline — Smoke Test")
    print("=" * 65 + "\n")

    # ----------------------------------------------------------------
    # Run the full pipeline.
    # On first run this will download ~23 MB of MIT-BIH data from
    # PhysioNet. Subsequent runs use the local cache.
    # ----------------------------------------------------------------
    loaders = build_mitbih_pipeline(
        data_dir='./mitbih_data',
        batch_size=BATCH_SIZE,
        num_workers=0,   # Use 0 workers for the smoke test (avoids
                         # multiprocessing overhead in __main__ on Windows)
        seed=RANDOM_SEED
    )

    # ----------------------------------------------------------------
    # Pull one batch from each loader and verify shapes & dtypes.
    # ----------------------------------------------------------------
    print("=" * 65)
    print("  Batch Shape Verification")
    print("=" * 65)

    for split_name, loader in loaders.items():
        signals, labels = next(iter(loader))

        print(f"\n  [{split_name.upper()} LOADER]")
        print(f"    Signal tensor shape  : {signals.shape}")
        print(f"    Label  tensor shape  : {labels.shape}")
        print(f"    Signal dtype         : {signals.dtype}")
        print(f"    Label  dtype         : {labels.dtype}")
        print(f"    Signal value range   : [{signals.min():.4f}, {signals.max():.4f}]")
        print(f"    Label unique classes : {labels.unique().tolist()}")

        # --- Assertion: critical shape check ---
        assert signals.ndim == 3, \
            f"Expected 3-D signal tensor (B, 1, 188), got {signals.ndim}-D"
        assert signals.shape[1] == 1, \
            f"Expected 1 channel, got {signals.shape[1]}"
        assert signals.shape[2] == SEGMENT_LEN, \
            f"Expected segment length {SEGMENT_LEN}, got {signals.shape[2]}"
        assert signals.dtype == torch.float32, \
            f"Expected float32, got {signals.dtype}"
        assert labels.dtype == torch.int64, \
            f"Expected int64, got {labels.dtype}"

    print("\n" + "=" * 65)
    print("  ✓ All assertions passed.")
    print(f"  ✓ Training batch shape confirmed: "
          f"[{BATCH_SIZE}, 1, {SEGMENT_LEN}]")
    print("=" * 65 + "\n")
