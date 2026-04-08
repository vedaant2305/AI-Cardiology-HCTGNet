# =============================================================================
# FILE: roc_auc.py
# PROJECT: Trustworthy Arrhythmia Diagnosis
# DESCRIPTION: Computes per-class ROC curves and AUC scores for HCTG-Net.
#              Produces a publication-quality figure for the IEEE paper.
#
# USAGE:
#   python roc_auc.py
#
# OUTPUT:
#   ./results/roc_curves.png       — per-class ROC curve figure
#   ./results/auc_scores.txt       — AUC scores for paper
#
# ASSUMES:
#   - best_hctg_net.pth exists in working directory
#   - model.py and preprocessing.py are in working directory
# =============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from preprocessing import build_mitbih_pipeline, CLASS_NAMES
from model import HCTGNet


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

CHECKPOINT_PATH = 'best_hctg_net.pth'
RESULTS_DIR     = './results'
NUM_CLASSES     = 5
SEED            = 42

# Colours for each class curve — matches clinical dashboard accent colours
CLASS_COLORS = {
    0: '#00c853',   # N — green
    1: '#ffd600',   # S — yellow
    2: '#ff6d00',   # V — orange
    3: '#aa00ff',   # F — purple
    4: '#00b4d8',   # Q — blue
}


# =============================================================================
# SECTION 2: DEVICE SETUP
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


# =============================================================================
# SECTION 3: COLLECT ALL PREDICTIONS FROM TEST SET
# =============================================================================

@torch.no_grad()
def get_predictions(
    model      : nn.Module,
    test_loader,
    device     : torch.device,
) -> tuple:
    """
    Runs the full test set through the model and collects:
        - all_probs   : softmax probabilities for every sample (N, 5)
        - all_targets : ground truth labels (N,)

    We need the FULL probability vector (not just argmax) for ROC curves.
    ROC curves require the confidence score for each class independently.

    Args:
        model       : Loaded HCTGNet in eval() mode
        test_loader : Test DataLoader
        device      : Compute device

    Returns:
        all_probs   (np.ndarray): Shape (N, 5)
        all_targets (np.ndarray): Shape (N,)
    """
    model.eval()

    all_probs   = []
    all_targets = []

    print("[INFERENCE] Running test set through model ...")

    for signals, labels in test_loader:
        signals = signals.to(device, non_blocking=True)

        logits = model(signals)                          # (B, 5)
        probs  = torch.softmax(logits, dim=1)            # (B, 5)

        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(labels.numpy())

    all_probs   = np.array(all_probs)                   # (N, 5)
    all_targets = np.array(all_targets)                 # (N,)

    print(f"  Collected {len(all_targets):,} predictions")
    return all_probs, all_targets


# =============================================================================
# SECTION 4: COMPUTE ROC CURVES AND AUC
# =============================================================================

def compute_roc_auc(
    all_probs  : np.ndarray,
    all_targets: np.ndarray,
    num_classes: int = 5,
) -> tuple:
    """
    Computes per-class ROC curves using One-vs-Rest (OvR) strategy.

    One-vs-Rest means:
        For Class V: treats V as positive, all other classes as negative
        For Class N: treats N as positive, all other classes as negative
        ... and so on for each class

    This is the standard approach for multi-class ROC analysis in
    medical AI papers.

    Args:
        all_probs   : Shape (N, 5) — softmax probabilities
        all_targets : Shape (N,)   — ground truth integer labels
        num_classes : Number of classes (5)

    Returns:
        fpr_dict  : dict {class_idx: fpr array}
        tpr_dict  : dict {class_idx: tpr array}
        auc_dict  : dict {class_idx: auc score}
        macro_auc : float — mean AUC across all classes
    """

    # Binarize labels for One-vs-Rest
    # e.g. label 2 (V) → [0, 0, 1, 0, 0]
    y_bin = label_binarize(all_targets, classes=list(range(num_classes)))

    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}

    print("\n[AUC] Per-class AUC Scores:")
    print(f"  {'Class':<20} {'AUC':>8}")
    print(f"  {'-'*30}")

    for i in range(num_classes):
        # Compute ROC curve for class i
        fpr, tpr, _ = roc_curve(
            y_bin[:, i],        # binary ground truth for class i
            all_probs[:, i],    # predicted probability for class i
        )
        roc_auc = auc(fpr, tpr)

        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        auc_dict[i] = roc_auc

        class_label = f"{CLASS_NAMES[i]} — {get_full_name(i)}"
        print(f"  {class_label:<20} {roc_auc:.4f}")

    # Macro AUC = simple average across all classes
    macro_auc = np.mean(list(auc_dict.values()))
    print(f"  {'-'*30}")
    print(f"  {'Macro AUC':<20} {macro_auc:.4f}")

    return fpr_dict, tpr_dict, auc_dict, macro_auc


def get_full_name(class_idx: int) -> str:
    """Returns the full clinical name for a class index."""
    names = {
        0: 'Normal',
        1: 'Supraventricular',
        2: 'Ventricular (PVC)',
        3: 'Fusion',
        4: 'Paced/Unknown',
    }
    return names[class_idx]


# =============================================================================
# SECTION 5: PLOT ROC CURVES
# =============================================================================

def plot_roc_curves(
    fpr_dict  : dict,
    tpr_dict  : dict,
    auc_dict  : dict,
    macro_auc : float,
    save_dir  : str,
):
    """
    Generates a publication-quality ROC curve figure with:
        - One coloured curve per class
        - AUC score in the legend for each class
        - Diagonal reference line (random classifier)
        - Macro AUC annotation
        - Dark background matching the project style

    Args:
        fpr_dict  : Per-class false positive rate arrays
        tpr_dict  : Per-class true positive rate arrays
        auc_dict  : Per-class AUC scores
        macro_auc : Mean AUC across all classes
        save_dir  : Directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # --- Plot each class curve ---
    for i in range(NUM_CLASSES):
        class_name = CLASS_NAMES[i]
        full_name  = get_full_name(i)
        color      = CLASS_COLORS[i]
        roc_auc    = auc_dict[i]

        ax.plot(
            fpr_dict[i],
            tpr_dict[i],
            color=color,
            linewidth=2.2,
            label=f"Class {class_name} ({full_name})  AUC = {roc_auc:.4f}",
            zorder=3,
        )

        # Shade area under the curve subtly
        ax.fill_between(
            fpr_dict[i], tpr_dict[i],
            alpha=0.04,
            color=color,
            zorder=2,
        )

    # --- Diagonal reference line (random classifier) ---
    ax.plot(
        [0, 1], [0, 1],
        color='#888888',
        linewidth=1.2,
        linestyle='--',
        label='Random Classifier  AUC = 0.5000',
        zorder=1,
    )

    # --- Macro AUC annotation box ---
    ax.text(
        0.58, 0.12,
        f"Macro AUC = {macro_auc:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        color='white',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='#2a2a4a',
            edgecolor='#555577',
            alpha=0.9,
        ),
        zorder=5,
    )

    # --- Axis formatting ---
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate (1 - Specificity)',
                  color='white', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)',
                  color='white', fontsize=11)
    ax.set_title(
        'HCTG-Net — Per-Class ROC Curves (One-vs-Rest)\nMIT-BIH Arrhythmia Database Test Set',
        color='white', fontsize=12, fontweight='bold', pad=14,
    )
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#333355')
    ax.grid(True, linestyle=':', alpha=0.2, color='white')

    # --- Legend ---
    legend = ax.legend(
        loc='lower right',
        fontsize=9,
        facecolor='#1a1a2e',
        edgecolor='#444466',
        labelcolor='white',
        framealpha=0.95,
    )

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(
        save_path, dpi=180,
        bbox_inches='tight',
        facecolor=fig.get_facecolor(),
    )
    plt.close()
    print(f"\n[PLOT] ROC curves saved → {save_path}")


# =============================================================================
# SECTION 6: SAVE AUC SCORES TO TEXT FILE
# =============================================================================

def save_auc_report(
    auc_dict  : dict,
    macro_auc : float,
    save_dir  : str,
):
    """
    Saves AUC scores to a text file for easy copy-paste into the paper.
    """
    lines = [
        "=" * 55,
        "  HCTG-Net — AUC Scores (IEEE Paper Table)",
        "=" * 55,
        "",
        f"  {'Class':<30} {'AUC':>8}",
        f"  {'-'*40}",
    ]

    for i in range(NUM_CLASSES):
        name = f"{CLASS_NAMES[i]} — {get_full_name(i)}"
        lines.append(f"  {name:<30} {auc_dict[i]:.4f}")

    lines += [
        f"  {'-'*40}",
        f"  {'Macro AUC (Mean)':<30} {macro_auc:.4f}",
        "",
        "  IEEE Reportable Result:",
        f"  The proposed HCTG-Net achieved a macro-average",
        f"  AUC of {macro_auc:.4f} across all 5 AAMI classes,",
        f"  evaluated on the MIT-BIH test set using the",
        f"  One-vs-Rest strategy.",
        "",
        "=" * 55,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    save_path = os.path.join(save_dir, 'auc_scores.txt')
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"\n[SAVED] AUC report → {save_path}")


# =============================================================================
# SECTION 7: ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("  HCTG-Net — ROC Curves and AUC Evaluation")
    print("=" * 55 + "\n")

    torch.manual_seed(SEED)
    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load best model
    # ------------------------------------------------------------------
    print(f"[MODEL] Loading checkpoint from '{CHECKPOINT_PATH}' ...")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    model = HCTGNet(
        num_classes=5,
        d_model=128,
        n_heads=4,
        ffn_dim=256,
        n_layers=2,
        dropout=0.1,
        clf_dropout=0.3,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded weights from epoch {checkpoint['epoch']}  "
          f"(val F1 = {checkpoint['val_f1']:.4f})")

    # ------------------------------------------------------------------
    # 2. Build test DataLoader
    # ------------------------------------------------------------------
    print("\n[DATA] Building test DataLoader ...")
    loaders = build_mitbih_pipeline(
        data_dir='./mitbih_data',
        batch_size=512,
        num_workers=0,
        seed=SEED,
    )
    test_loader = loaders['test']

    # ------------------------------------------------------------------
    # 3. Collect all predictions
    # ------------------------------------------------------------------
    all_probs, all_targets = get_predictions(model, test_loader, device)

    # ------------------------------------------------------------------
    # 4. Compute ROC curves and AUC
    # ------------------------------------------------------------------
    fpr_dict, tpr_dict, auc_dict, macro_auc = compute_roc_auc(
        all_probs, all_targets, NUM_CLASSES
    )

    # ------------------------------------------------------------------
    # 5. Plot and save
    # ------------------------------------------------------------------
    plot_roc_curves(
        fpr_dict, tpr_dict, auc_dict,
        macro_auc, RESULTS_DIR
    )

    save_auc_report(auc_dict, macro_auc, RESULTS_DIR)

    # ------------------------------------------------------------------
    # 6. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  DONE! Files saved:")
    print(f"  ./results/roc_curves.png")
    print(f"  ./results/auc_scores.txt")
    print("=" * 55)
    print(f"\n  IEEE Result: Macro AUC = {macro_auc:.4f}")

    if macro_auc >= 0.99:
        print("  ✅ Excellent — AUC ≥ 0.99, publication ready!")
    elif macro_auc >= 0.97:
        print("  ✅ Very strong — AUC ≥ 0.97, publication ready!")
    else:
        print("  ⚠️  AUC below 0.97 — consider more training epochs")
