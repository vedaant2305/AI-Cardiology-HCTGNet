# =============================================================================
# FILE: train.py
# PROJECT: Trustworthy Arrhythmia Diagnosis
# DESCRIPTION: Complete training, validation, and final test-set evaluation
#              pipeline for HCTG-Net on the MIT-BIH Arrhythmia Database.
#
#              Covers:
#                - Device-agnostic setup (CUDA / MPS / CPU)
#                - CrossEntropyLoss + Adam + ReduceLROnPlateau scheduler
#                - Per-epoch Loss, Accuracy, and Macro F1-Score
#                - Best-model checkpointing on validation F1
#                - Final test evaluation with classification report
#                  and confusion matrix
#
# USAGE:
#   python train.py
#
# DEPENDENCIES (beyond preprocessing.py and model.py):
#   pip install scikit-learn matplotlib seaborn
# =============================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import build_mitbih_pipeline, CLASS_NAMES
from model import HCTGNet



# =============================================================================
# SECTION 1: CONFIGURATION DATACLASS
# =============================================================================

class TrainConfig:
    """
    Central configuration object. Edit values here rather than hunting
    through the code for magic numbers.
    """
    # --- Data ---
    data_dir    : str   = './mitbih_data'
    batch_size  : int   = 256
    num_workers : int   = 4

    # --- Model ---
    num_classes : int   = 5
    d_model     : int   = 128
    n_heads     : int   = 4
    ffn_dim     : int   = 256
    n_layers    : int   = 2
    dropout     : float = 0.1
    clf_dropout : float = 0.3

    # --- Optimiser ---
    learning_rate : float = 1e-3
    weight_decay  : float = 1e-4

    # --- Scheduler (ReduceLROnPlateau) ---
    # Reduces LR by factor when val F1 does not improve for `patience` epochs.
    lr_factor   : float = 0.5
    lr_patience : int   = 5
    lr_min      : float = 1e-6

    # --- Training ---
    num_epochs      : int   = 5      # set to 100 for a full production run
    grad_clip_norm  : float = 1.0    # gradient clipping max-norm (paper §3.1)
    seed            : int   = 42

    # --- Checkpointing & Outputs ---
    checkpoint_path : str = 'best_hctg_net.pth'
    results_dir     : str = './results'   # confusion matrix PNG saved here


# =============================================================================
# SECTION 2: DEVICE SETUP
# =============================================================================

def get_device() -> torch.device:
    """
    Returns the best available compute device in priority order:
        1. CUDA  (NVIDIA GPU)
        2. MPS   (Apple Silicon GPU via Metal Performance Shaders)
        3. CPU   (fallback)

    The returned device object is passed to .to(device) calls throughout.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[DEVICE] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("[DEVICE] Apple MPS (Metal) GPU detected.")
    else:
        device = torch.device('cpu')
        print("[DEVICE] No GPU found — running on CPU.")
    return device


# =============================================================================
# SECTION 3: SEED EVERYTHING FOR REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42):
    """
    Sets all relevant random seeds so that training runs are reproducible.
    Covers Python, NumPy, PyTorch CPU, and PyTorch CUDA (if available).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic CUDA ops where available (may slow down training
    # slightly; remove if speed is more important than strict reproducibility).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# SECTION 4: SINGLE EPOCH — TRAINING PASS
# =============================================================================

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
    grad_clip : float,
) -> dict:
    """
    Runs one full pass over the training DataLoader, updating model weights.

    Steps per batch:
        1. Move data to device
        2. Zero gradients
        3. Forward pass
        4. Compute CrossEntropy loss
        5. Backward pass
        6. Gradient clipping (prevents exploding gradients)
        7. Optimiser step
        8. Accumulate metrics

    Args:
        model     : HCTG-Net instance (in training mode).
        loader    : Training DataLoader.
        criterion : nn.CrossEntropyLoss instance.
        optimizer : Adam optimiser.
        device    : Target compute device.
        grad_clip : Max-norm for gradient clipping.

    Returns:
        dict with keys 'loss', 'accuracy', 'f1' for this epoch.
    """
    model.train()   # activates Dropout and BatchNorm training behaviour

    running_loss   = 0.0
    all_preds      = []
    all_targets    = []

    for batch_idx, (signals, labels) in enumerate(loader):

        # --- Move tensors to the target device (GPU / MPS / CPU) ---
        signals = signals.to(device, non_blocking=True)  # (B, 1, 188)
        labels  = labels.to(device,  non_blocking=True)  # (B,)

        # --- Zero the parameter gradients ---
        # set_to_none=True is slightly faster than .zero_grad() as it
        # deallocates gradient memory rather than zeroing it in-place.
        optimizer.zero_grad(set_to_none=True)

        # --- Forward pass ---
        logits = model(signals)   # (B, 5) — raw class scores

        # --- Compute loss ---
        # nn.CrossEntropyLoss applies log-softmax internally, so logits
        # (not probabilities) are passed directly.
        loss = criterion(logits, labels)

        # --- Backward pass ---
        loss.backward()

        # --- Gradient clipping ---
        # Clips the global L2-norm of all gradients to `grad_clip`.
        # Prevents a single bad batch from causing a catastrophic weight update.
        # The paper uses max_norm=1.0 (Section 3.1).
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # --- Parameter update ---
        optimizer.step()

        # --- Accumulate batch metrics ---
        running_loss += loss.item() * signals.size(0)  # un-average the loss

        # Move predictions to CPU for sklearn metrics (avoids GPU memory build-up)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    # --- Compute epoch-level metrics ---
    n_samples     = len(loader.dataset)
    epoch_loss    = running_loss / n_samples
    epoch_acc     = accuracy_score(all_targets, all_preds)
    epoch_f1      = f1_score(all_targets, all_preds,
                             average='macro', zero_division=0)

    return {'loss': epoch_loss, 'accuracy': epoch_acc, 'f1': epoch_f1}


# =============================================================================
# SECTION 5: SINGLE EPOCH — VALIDATION / TEST PASS
# =============================================================================

def evaluate(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    device    : torch.device,
) -> dict:
    """
    Runs one full pass over a DataLoader WITHOUT updating weights.
    Used for both validation (each epoch) and final test evaluation.

    Key differences from train_one_epoch:
        - model.eval()           : disables Dropout, freezes BatchNorm stats
        - torch.no_grad()        : disables gradient graph — saves memory & time
        - No optimizer.step()    : weights are NOT updated

    Args:
        model     : HCTG-Net instance.
        loader    : Validation or Test DataLoader.
        criterion : nn.CrossEntropyLoss instance.
        device    : Target compute device.

    Returns:
        dict with keys 'loss', 'accuracy', 'f1', 'preds', 'targets'.
        'preds' and 'targets' are returned for the final confusion matrix.
    """
    model.eval()   # disables dropout; BatchNorm uses running statistics

    running_loss = 0.0
    all_preds    = []
    all_targets  = []

    with torch.no_grad():   # no gradient tracking needed during evaluation
        for signals, labels in loader:

            signals = signals.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            logits  = model(signals)                  # (B, 5)
            loss    = criterion(logits, labels)

            running_loss += loss.item() * signals.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    n_samples  = len(loader.dataset)
    epoch_loss = running_loss / n_samples
    epoch_acc  = accuracy_score(all_targets, all_preds)
    epoch_f1   = f1_score(all_targets, all_preds,
                          average='macro', zero_division=0)

    return {
        'loss'   : epoch_loss,
        'accuracy': epoch_acc,
        'f1'     : epoch_f1,
        'preds'  : np.array(all_preds),
        'targets': np.array(all_targets),
    }


# =============================================================================
# SECTION 6: LEARNING CURVE PLOTTING
# =============================================================================

def plot_learning_curves(history: dict, save_dir: str):
    """
    Saves a 3-panel figure (Loss / Accuracy / F1-Score) showing training
    and validation curves across epochs.

    Args:
        history  : dict produced by the training loop containing lists:
                   'train_loss', 'val_loss', 'train_acc', 'val_acc',
                   'train_f1', 'val_f1'.
        save_dir : Directory where the PNG will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('HCTG-Net Training Curves', fontsize=14, fontweight='bold')

    panels = [
        ('Loss',       'train_loss', 'val_loss'),
        ('Accuracy',   'train_acc',  'val_acc'),
        ('Macro F1',   'train_f1',   'val_f1'),
    ]

    for ax, (title, train_key, val_key) in zip(axes, panels):
        ax.plot(epochs, history[train_key], 'b-o', markersize=3,
                label='Train', linewidth=1.5)
        ax.plot(epochs, history[val_key],   'r-o', markersize=3,
                label='Val',   linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'learning_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Learning curves saved → {save_path}")


# =============================================================================
# SECTION 7: CONFUSION MATRIX PLOTTING
# =============================================================================

def plot_confusion_matrix(
    targets  : np.ndarray,
    preds    : np.ndarray,
    save_dir : str,
):
    """
    Generates and saves a normalised confusion matrix heatmap using
    matplotlib + seaborn. Values are row-normalised (recall per class).

    Args:
        targets  : Ground-truth integer labels (N,).
        preds    : Predicted integer labels (N,).
        save_dir : Directory where the PNG will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build confusion matrix and normalise each row to [0, 1]
    cm = confusion_matrix(targets, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    class_labels = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title('HCTG-Net — Normalised Confusion Matrix (Test Set)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label',      fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Confusion matrix saved → {save_path}")


# =============================================================================
# SECTION 8: MASTER TRAINING LOOP
# =============================================================================

def train(config: TrainConfig):
    """
    Orchestrates the full training pipeline:
        1.  Set seeds for reproducibility
        2.  Detect compute device
        3.  Build DataLoaders via the preprocessing pipeline
        4.  Instantiate model, loss, optimiser, scheduler
        5.  Run training loop for config.num_epochs epochs
        6.  Checkpoint the model whenever val F1 improves
        7.  After training, load best weights and evaluate on the test set
        8.  Print classification report and save confusion matrix

    Args:
        config: TrainConfig instance holding all hyperparameters.
    """

    # -------------------------------------------------------------------------
    # 1. Reproducibility & device
    # -------------------------------------------------------------------------
    set_seed(config.seed)
    device = get_device()
    os.makedirs(config.results_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data
    # -------------------------------------------------------------------------
    print("\n[DATA] Building MIT-BIH DataLoaders ...")
    loaders = build_mitbih_pipeline(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    train_loader = loaders['train']
    val_loader   = loaders['val']
    test_loader  = loaders['test']

    # -------------------------------------------------------------------------
    # 3. Model
    # -------------------------------------------------------------------------
    print("\n[MODEL] Initialising HCTG-Net ...")
    model = HCTGNet(
        num_classes=config.num_classes,
        d_model=config.d_model,
        n_heads=config.n_heads,
        ffn_dim=config.ffn_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        clf_dropout=config.clf_dropout,
    ).to(device)

    print(f"  Trainable parameters: {model.count_parameters():,}")

    # -------------------------------------------------------------------------
    # 4. Loss function
    # -------------------------------------------------------------------------
    # nn.CrossEntropyLoss combines LogSoftmax + NLLLoss in a single numerically
    # stable operation. Use class weighting to further encourage macro-F1 gains.
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(config.num_classes),
        y=np.concatenate([batch[1].numpy() for batch in train_loader])
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"\n[LOSS] Class weights: {{ {', '.join(f'{i}: {w:.3f}' for i, w in enumerate(class_weights))} }}")
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # -------------------------------------------------------------------------
    # 5. Optimiser
    # -------------------------------------------------------------------------
    # Adam with weight_decay acts as L2 regularisation on the weights.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # -------------------------------------------------------------------------
    # 6. Learning rate scheduler
    # -------------------------------------------------------------------------
    # ReduceLROnPlateau monitors val F1 (higher = better, so mode='max').
    # If F1 does not improve for `patience` epochs, LR is multiplied by `factor`.
    # This is more adaptive than cosine annealing for variable-length runs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',                 # we want F1 to go UP
        factor=config.lr_factor,    # multiply LR by this on plateau
        patience=config.lr_patience,
        min_lr=config.lr_min
                     # prints a message when LR is reduced
    )

    # -------------------------------------------------------------------------
    # 7. Training loop state
    # -------------------------------------------------------------------------
    best_val_f1  = -1.0   # sentinel: any real F1 will beat this
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : [],
        'train_f1'  : [], 'val_f1'  : [],
    }

    print(f"\n[TRAIN] Starting training for {config.num_epochs} epoch(s) ...")
    print(f"  Checkpoint path : {config.checkpoint_path}")
    print(f"  Results dir     : {config.results_dir}")
    print("=" * 72)

    total_train_start = time.time()

    for epoch in range(1, config.num_epochs + 1):

        epoch_start = time.time()

        # --- Training pass ---
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=config.grad_clip_norm,
        )

        # --- Validation pass ---
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # --- Step the LR scheduler based on val F1 ---
        # ReduceLROnPlateau needs the monitored metric, not the epoch number.
        scheduler.step(val_metrics['f1'])

        # --- Log metrics ---
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'  ].append(val_metrics['loss'])
        history['train_acc' ].append(train_metrics['accuracy'])
        history['val_acc'   ].append(val_metrics['accuracy'])
        history['train_f1'  ].append(train_metrics['f1'])
        history['val_f1'    ].append(val_metrics['f1'])

        current_lr = optimizer.param_groups[0]['lr']

        # --- Console output ---
        print(
            f"  Epoch [{epoch:>3}/{config.num_epochs}]  "
            f"T-Loss: {train_metrics['loss']:.4f}  "
            f"T-F1: {train_metrics['f1']:.4f}  |  "
            f"V-Loss: {val_metrics['loss']:.4f}  "
            f"V-F1: {val_metrics['f1']:.4f}  "
            f"V-Acc: {val_metrics['accuracy']:.4f}  |  "
            f"LR: {current_lr:.2e}  "
            f"[{epoch_time:.1f}s]"
        )

        # -----------------------------------------------------------------
        # 8. Model checkpointing — save whenever val F1 improves
        # -----------------------------------------------------------------
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']

            # Save the full checkpoint: weights + optimiser state + metadata.
            # Saving the optimiser state allows training to be resumed exactly
            # from this point if needed.
            checkpoint = {
                'epoch'           : epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'val_f1'          : best_val_f1,
                'val_loss'        : val_metrics['loss'],
                'config'          : config.__dict__,
            }
            torch.save(checkpoint, config.checkpoint_path)
            print(f"    ✓ New best val F1 = {best_val_f1:.4f} "
                  f"— checkpoint saved → {config.checkpoint_path}")

    total_train_time = time.time() - total_train_start
    print("=" * 72)
    print(f"[TRAIN] Finished. Total time: {total_train_time/60:.1f} min  |  "
          f"Best val F1: {best_val_f1:.4f}\n")

    # -------------------------------------------------------------------------
    # 9. Save learning curves
    # -------------------------------------------------------------------------
    plot_learning_curves(history, config.results_dir)

    # =========================================================================
    # SECTION 9: FINAL TEST SET EVALUATION
    # =========================================================================
    # Load the BEST checkpoint (not the last epoch's weights — these may have
    # overfit if training ran many epochs past the best val F1 point).
    print(f"[TEST] Loading best checkpoint from '{config.checkpoint_path}' ...")
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded weights from epoch {checkpoint['epoch']}  "
          f"(val F1 = {checkpoint['val_f1']:.4f})\n")

    # Run inference on the completely held-out test set
    test_metrics = evaluate(model, test_loader, criterion, device)

    # -------------------------------------------------------------------------
    # 10. Print classification report
    # -------------------------------------------------------------------------
    class_labels = [CLASS_NAMES[i] for i in range(config.num_classes)]

    print("=" * 72)
    print("  FINAL TEST SET RESULTS")
    print("=" * 72)
    print(f"  Test Loss     : {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro F1 : {test_metrics['f1']:.4f}")
    print()
    print("  Per-Class Classification Report:")
    print("-" * 72)
    print(
        classification_report(
            test_metrics['targets'],
            test_metrics['preds'],
            target_names=class_labels,
            digits=4,
            zero_division=0,
        )
    )
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 11. Confusion matrix
    # -------------------------------------------------------------------------
    plot_confusion_matrix(
        test_metrics['targets'],
        test_metrics['preds'],
        save_dir=config.results_dir,
    )

    return history, test_metrics


# =============================================================================
# SECTION 10: ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Instantiate configuration.
    # num_epochs=5 is set here for a fast smoke-test run.
    # For a full production run, set num_epochs=100 (as per the paper).
    # ------------------------------------------------------------------
    cfg = TrainConfig()
    cfg.num_epochs  = 100    # ← change to 100 for full training
    cfg.num_workers = 4    # ← set to 0 for Windows / debugging; 4 for Linux

    print("=" * 72)
    print("  Trustworthy Arrhythmia Diagnosis — HCTG-Net Training")
    print("=" * 72)
    print(f"  Epochs      : {cfg.num_epochs}")
    print(f"  Batch size  : {cfg.batch_size}")
    print(f"  LR          : {cfg.learning_rate}")
    print(f"  Checkpoint  : {cfg.checkpoint_path}")
    print(f"  Results dir : {cfg.results_dir}")
    print("=" * 72 + "\n")

    history, test_metrics = train(cfg)

    print("\n[DONE] Training and evaluation complete.")
    print(f"  Best checkpoint : {cfg.checkpoint_path}")
    print(f"  Plots saved in  : {cfg.results_dir}/")
