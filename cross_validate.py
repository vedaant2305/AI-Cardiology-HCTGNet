# =============================================================================
# FILE: cross_validate.py
# PROJECT: Trustworthy Arrhythmia Diagnosis
# DESCRIPTION: 5-Fold Stratified Cross Validation for HCTG-Net.
#              Produces mean ± std across all folds for IEEE-level reporting.
#
# USAGE:
#   python cross_validate.py
#
# OUTPUT:
#   ./results/crossval/
#       fold_1_model.pth  ... fold_5_model.pth   — best model per fold
#       fold_1_report.txt ... fold_5_report.txt   — classification report
#       crossval_summary.txt                       — final mean ± std table
#       crossval_metrics.png                       — bar chart of fold results
# =============================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocessing import (
    download_and_load_mitbih,
    z_score_normalise,
    apply_smote,
    ECGDataset,
    RANDOM_SEED,
    CLASS_NAMES,
)
from model import HCTGNet


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

class CVConfig:
    # --- Data ---
    data_dir     : str   = './mitbih_data'

    # --- Cross Validation ---
    n_folds      : int   = 5
    epochs       : int   = 10       # 10 epochs per fold — quick version
    batch_size   : int   = 512
    num_workers  : int   = 0        # 0 for Windows

    # --- Model ---
    num_classes  : int   = 5
    d_model      : int   = 128
    n_heads      : int   = 4
    ffn_dim      : int   = 256
    n_layers     : int   = 2
    dropout      : float = 0.1
    clf_dropout  : float = 0.3

    # --- Optimiser ---
    learning_rate : float = 1e-3
    weight_decay  : float = 1e-4
    grad_clip     : float = 1.0

    # --- Scheduler ---
    lr_factor    : float = 0.5
    lr_patience  : int   = 3
    lr_min       : float = 1e-6

    # --- Output ---
    results_dir  : str   = './results/crossval'
    seed         : int   = RANDOM_SEED


# =============================================================================
# SECTION 2: DEVICE + SEED
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[DEVICE] CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("[DEVICE] Apple MPS GPU")
    else:
        device = torch.device('cpu')
        print("[DEVICE] CPU")
    return device


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# SECTION 3: TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    running_loss = 0.0
    all_preds    = []
    all_targets  = []

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

    n          = len(loader.dataset)
    epoch_loss = running_loss / n
    epoch_f1   = f1_score(all_targets, all_preds,
                          average='macro', zero_division=0)
    epoch_acc  = accuracy_score(all_targets, all_preds)
    return {'loss': epoch_loss, 'f1': epoch_f1, 'accuracy': epoch_acc}


# =============================================================================
# SECTION 4: EVALUATE
# =============================================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds    = []
    all_targets  = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)
            logits  = model(signals)
            loss    = criterion(logits, labels)
            running_loss += loss.item() * signals.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    n          = len(loader.dataset)
    epoch_loss = running_loss / n
    epoch_f1   = f1_score(all_targets, all_preds,
                          average='macro', zero_division=0)
    epoch_acc  = accuracy_score(all_targets, all_preds)
    return {
        'loss'   : epoch_loss,
        'f1'     : epoch_f1,
        'accuracy': epoch_acc,
        'preds'  : np.array(all_preds),
        'targets': np.array(all_targets),
    }


# =============================================================================
# SECTION 5: TRAIN ONE FOLD
# =============================================================================

def train_fold(
    fold_idx  : int,
    X_train   : np.ndarray,
    y_train   : np.ndarray,
    X_val     : np.ndarray,
    y_val     : np.ndarray,
    config    : CVConfig,
    device    : torch.device,
) -> dict:
    """
    Trains HCTG-Net for one cross-validation fold.

    Steps:
        1. Apply SMOTE to training fold only
        2. Build DataLoaders
        3. Train for config.epochs epochs
        4. Save best model (by val F1) for this fold
        5. Return metrics

    Args:
        fold_idx : Current fold number (1-indexed for display)
        X_train  : Training segments for this fold
        y_train  : Training labels for this fold
        X_val    : Validation segments for this fold
        y_val    : Validation labels for this fold
        config   : CVConfig instance
        device   : Compute device

    Returns:
        dict of best val metrics for this fold
    """

    print(f"\n{'='*65}")
    print(f"  FOLD {fold_idx} / {config.n_folds}")
    print(f"{'='*65}")
    print(f"  Train: {len(X_train):,} samples  |  Val: {len(X_val):,} samples")

    # ------------------------------------------------------------------
    # 1. SMOTE on training fold ONLY
    # ------------------------------------------------------------------
    print(f"  Applying SMOTE to fold {fold_idx} training set ...")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train, seed=config.seed)

    # ------------------------------------------------------------------
    # 2. Build DataLoaders
    # ------------------------------------------------------------------
    train_dataset = ECGDataset(X_train_sm, y_train_sm)
    val_dataset   = ECGDataset(X_val,      y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 3. Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model = HCTGNet(
        num_classes=config.num_classes,
        d_model=config.d_model,
        n_heads=config.n_heads,
        ffn_dim=config.ffn_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        clf_dropout=config.clf_dropout,
    ).to(device)

    # Class weights from resampled training labels
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(config.num_classes),
        y=y_train_sm,
    )
    weights_tensor = torch.tensor(
        class_weights, dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.lr_min,
    )

    # ------------------------------------------------------------------
    # 4. Training loop for this fold
    # ------------------------------------------------------------------
    best_val_f1   = -1.0
    best_metrics  = {}
    best_preds    = None
    best_targets  = None

    checkpoint_path = os.path.join(
        config.results_dir, f'fold_{fold_idx}_model.pth'
    )

    print(f"\n  Training for {config.epochs} epochs ...")
    print(f"  {'Epoch':<8} {'T-Loss':<10} {'T-F1':<10} "
          f"{'V-Loss':<10} {'V-F1':<10} {'V-Acc':<10} {'Time'}")
    print(f"  {'-'*62}")

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, criterion,
            optimizer, device, config.grad_clip
        )
        val_m = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_m['f1'])
        elapsed = time.time() - t0

        print(
            f"  [{epoch:>2}/{config.epochs}]  "
            f"{train_m['loss']:.4f}     "
            f"{train_m['f1']:.4f}     "
            f"{val_m['loss']:.4f}     "
            f"{val_m['f1']:.4f}     "
            f"{val_m['accuracy']:.4f}     "
            f"[{elapsed:.0f}s]"
        )

        # Save best model for this fold
        if val_m['f1'] > best_val_f1:
            best_val_f1  = val_m['f1']
            best_metrics = {
                'fold'    : fold_idx,
                'f1'      : val_m['f1'],
                'accuracy': val_m['accuracy'],
                'loss'    : val_m['loss'],
                'epoch'   : epoch,
            }
            best_preds   = val_m['preds']
            best_targets = val_m['targets']

            torch.save({
                'fold'            : fold_idx,
                'epoch'           : epoch,
                'model_state_dict': model.state_dict(),
                'val_f1'          : best_val_f1,
            }, checkpoint_path)

    print(f"\n  ✓ Fold {fold_idx} best val F1 = {best_val_f1:.4f} "
          f"(epoch {best_metrics['epoch']})")

    # ------------------------------------------------------------------
    # 5. Save classification report for this fold
    # ------------------------------------------------------------------
    class_labels  = [CLASS_NAMES[i] for i in range(config.num_classes)]
    report        = classification_report(
        best_targets, best_preds,
        target_names=class_labels,
        digits=4,
        zero_division=0,
    )

    report_path = os.path.join(
        config.results_dir, f'fold_{fold_idx}_report.txt'
    )
    with open(report_path, 'w') as f:
        f.write(f"FOLD {fold_idx} — Best Epoch: {best_metrics['epoch']}\n")
        f.write(f"Val F1: {best_val_f1:.4f}  |  "
                f"Val Acc: {best_metrics['accuracy']:.4f}\n\n")
        f.write(report)

    print(f"  Report saved → {report_path}")

    # Free GPU memory before next fold
    del model
    torch.cuda.empty_cache()

    return best_metrics


# =============================================================================
# SECTION 6: SUMMARY PLOT
# =============================================================================

def plot_fold_results(fold_metrics: list, save_dir: str):
    """
    Saves a bar chart comparing F1 and Accuracy across all folds,
    with a horizontal line showing the mean.
    """
    folds     = [f"Fold {m['fold']}" for m in fold_metrics]
    f1_scores = [m['f1']       * 100 for m in fold_metrics]
    acc_scores= [m['accuracy'] * 100 for m in fold_metrics]

    mean_f1   = np.mean(f1_scores)
    std_f1    = np.std(f1_scores)
    mean_acc  = np.mean(acc_scores)
    std_acc   = np.std(acc_scores)

    x     = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars1 = ax.bar(x - width/2, f1_scores,  width,
                   label='Macro F1 (%)',  color='#00b4d8', alpha=0.85)
    bars2 = ax.bar(x + width/2, acc_scores, width,
                   label='Accuracy (%)', color='#90e0ef', alpha=0.85)

    # Mean lines
    ax.axhline(mean_f1,  color='#00b4d8', linestyle='--',
               linewidth=1.5, alpha=0.7,
               label=f'Mean F1  = {mean_f1:.2f}% ± {std_f1:.2f}%')
    ax.axhline(mean_acc, color='#90e0ef', linestyle='--',
               linewidth=1.5, alpha=0.7,
               label=f'Mean Acc = {mean_acc:.2f}% ± {std_acc:.2f}%')

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{bar.get_height():.2f}%',
                ha='center', va='bottom',
                color='white', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{bar.get_height():.2f}%',
                ha='center', va='bottom',
                color='white', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(folds, color='white', fontsize=10)
    ax.set_ylabel('Score (%)', color='white', fontsize=11)
    ax.set_ylim(min(f1_scores + acc_scores) - 3, 101)
    ax.set_title('HCTG-Net — 5-Fold Cross Validation Results',
                 color='white', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333355')
    ax.grid(axis='y', linestyle=':', alpha=0.3, color='white')
    ax.legend(fontsize=9, facecolor='#1a1a2e',
              edgecolor='#444466', labelcolor='white')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'crossval_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[PLOT] Cross-validation chart saved → {save_path}")


# =============================================================================
# SECTION 7: MASTER CROSS VALIDATION FUNCTION
# =============================================================================

def run_cross_validation(config: CVConfig):
    """
    Orchestrates the full 5-fold stratified cross-validation:
        1. Load and normalise the full dataset
        2. Create stratified folds (NO SMOTE yet — applied per fold)
        3. Train each fold independently
        4. Aggregate results and print mean ± std
        5. Save summary and plots
    """

    set_seed(config.seed)
    device = get_device()
    os.makedirs(config.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load full dataset (no split — CV handles splitting)
    # ------------------------------------------------------------------
    print("\n[DATA] Loading full MIT-BIH dataset ...")
    segments, labels = download_and_load_mitbih(config.data_dir)
    segments = z_score_normalise(segments)
    print(f"  Total samples: {len(segments):,}")

    # ------------------------------------------------------------------
    # 2. Stratified K-Fold splitter
    # ------------------------------------------------------------------
    # Stratified = each fold has the same class proportions as the
    # full dataset. This is essential for imbalanced medical data.
    skf = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.seed,
    )

    # ------------------------------------------------------------------
    # 3. Train each fold
    # ------------------------------------------------------------------
    all_fold_metrics = []

    fold_start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(segments, labels), start=1
    ):
        X_train, X_val = segments[train_idx], segments[val_idx]
        y_train, y_val = labels[train_idx],   labels[val_idx]

        fold_metrics = train_fold(
            fold_idx=fold_idx,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
            device=device,
        )
        all_fold_metrics.append(fold_metrics)

    total_time = (time.time() - fold_start_time) / 60

    # ------------------------------------------------------------------
    # 4. Aggregate results
    # ------------------------------------------------------------------
    f1_scores  = np.array([m['f1']       for m in all_fold_metrics])
    acc_scores = np.array([m['accuracy'] for m in all_fold_metrics])

    mean_f1  = f1_scores.mean()
    std_f1   = f1_scores.std()
    mean_acc = acc_scores.mean()
    std_acc  = acc_scores.std()

    # ------------------------------------------------------------------
    # 5. Print and save summary
    # ------------------------------------------------------------------
    summary_lines = [
        "=" * 65,
        "  HCTG-Net — 5-Fold Cross Validation Summary",
        "=" * 65,
        "",
        f"  {'Fold':<8} {'Val F1':>10} {'Val Accuracy':>14}",
        f"  {'-'*35}",
    ]
    for m in all_fold_metrics:
        summary_lines.append(
            f"  Fold {m['fold']:<4} {m['f1']*100:>9.4f}%"
            f" {m['accuracy']*100:>13.4f}%"
        )

    summary_lines += [
        f"  {'-'*35}",
        f"  {'Mean':<8} {mean_f1*100:>9.4f}%  {mean_acc*100:>13.4f}%",
        f"  {'Std':<8} {std_f1*100:>9.4f}%  {std_acc*100:>13.4f}%",
        "",
        "  IEEE Reportable Result:",
        f"  Macro F1  : {mean_f1*100:.2f}% ± {std_f1*100:.2f}%",
        f"  Accuracy  : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%",
        "",
        f"  Total training time: {total_time:.1f} minutes",
        "=" * 65,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # Save to file
    summary_path = os.path.join(config.results_dir, 'crossval_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"\n[SAVED] Summary → {summary_path}")

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    plot_fold_results(all_fold_metrics, config.results_dir)

    return all_fold_metrics, mean_f1, std_f1


# =============================================================================
# SECTION 8: ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  HCTG-Net — 5-Fold Stratified Cross Validation")
    print("=" * 65)

    cfg = CVConfig()

    print(f"\n  Folds        : {cfg.n_folds}")
    print(f"  Epochs/fold  : {cfg.epochs}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Results dir  : {cfg.results_dir}")
    print(f"\n  Estimated time: ~{cfg.n_folds * cfg.epochs * 2.8 / 60:.1f} hours")
    print("  Starting now — you can leave this running.\n")

    fold_metrics, mean_f1, std_f1 = run_cross_validation(cfg)

    print("\n[DONE] Cross validation complete!")
    print(f"  Final IEEE Result: Macro F1 = "
          f"{mean_f1*100:.2f}% ± {std_f1*100:.2f}%")
    print(f"  All files saved in: {cfg.results_dir}/")
