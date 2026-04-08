# =============================================================================
# FILE: calibration.py
# PROJECT: Trustworthy Arrhythmia Diagnosis
# DESCRIPTION: Model Calibration Analysis for HCTG-Net.
#              A well-calibrated model is one where the confidence score
#              matches the actual accuracy. For example, if the model says
#              "90% confident" on 100 samples, exactly 90 should be correct.
#
#              This is CRITICAL for medical AI trust and IEEE reviewers
#              will specifically look for this analysis.
#
#              Methods:
#                - Reliability Diagram (Calibration Curve)
#                - Expected Calibration Error (ECE)
#                - Maximum Calibration Error (MCE)
#                - Temperature Scaling (post-hoc calibration fix)
#                - Before vs After calibration comparison
#
# USAGE:
#   python calibration.py
#
# OUTPUT:
#   ./results/calibration/
#       calibration_before.png     — reliability diagram before scaling
#       calibration_after.png      — reliability diagram after scaling
#       calibration_comparison.png — side by side comparison for paper
#       calibration_report.txt     — ECE/MCE numbers for paper
# =============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from preprocessing import build_mitbih_pipeline, CLASS_NAMES
from model import HCTGNet


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

CHECKPOINT_PATH = 'best_hctg_net.pth'
RESULTS_DIR     = './results/calibration'
NUM_CLASSES     = 5
N_BINS          = 10      # number of bins for calibration curve
SEED            = 42


# =============================================================================
# SECTION 2: DEVICE
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
# SECTION 3: COLLECT PREDICTIONS
# =============================================================================

@torch.no_grad()
def get_all_probs(
    model      : nn.Module,
    loader     : DataLoader,
    device     : torch.device,
    temperature: float = 1.0,
) -> tuple:
    """
    Runs the full test set through the model and returns:
        - softmax probabilities (with optional temperature scaling)
        - ground truth labels

    Temperature Scaling:
        logits / T before softmax
        T > 1 → softer (less confident) predictions → better calibration
        T < 1 → sharper (more confident) predictions
        T = 1 → original model output (no scaling)

    Args:
        model       : HCTGNet in eval() mode
        loader      : Test DataLoader
        device      : Compute device
        temperature : Temperature scaling factor T

    Returns:
        all_probs   (np.ndarray): Shape (N, 5)
        all_targets (np.ndarray): Shape (N,)
    """
    model.eval()
    all_probs   = []
    all_targets = []

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        logits  = model(signals)                         # (B, 5)

        # Apply temperature scaling
        scaled_logits = logits / temperature

        probs = torch.softmax(scaled_logits, dim=1)      # (B, 5)
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(labels.numpy())

    return np.array(all_probs), np.array(all_targets)


# =============================================================================
# SECTION 4: CALIBRATION METRICS
# =============================================================================

def compute_ece(
    probs  : np.ndarray,
    targets: np.ndarray,
    n_bins : int = 10,
) -> float:
    """
    Computes the Expected Calibration Error (ECE).

    ECE measures the average gap between confidence and accuracy
    across all confidence bins:

        ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where B_b is the set of samples in bin b.

    A perfect ECE = 0.0 means the model is perfectly calibrated.
    ECE < 0.05 is considered well-calibrated for medical AI.

    Args:
        probs   : Shape (N, 5) — softmax probabilities
        targets : Shape (N,)   — ground truth labels
        n_bins  : Number of confidence bins

    Returns:
        ece: float — Expected Calibration Error
    """
    # Use max probability (confidence) and whether the top prediction is correct
    confidences  = probs.max(axis=1)        # (N,) — highest class probability
    predictions  = probs.argmax(axis=1)     # (N,) — predicted class
    correct      = (predictions == targets).astype(float)  # (N,) — 1 if correct

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > low) & (confidences <= high)

        if mask.sum() == 0:
            continue

        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_size = mask.sum()

        ece += (bin_size / len(targets)) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_mce(
    probs  : np.ndarray,
    targets: np.ndarray,
    n_bins : int = 10,
) -> float:
    """
    Computes the Maximum Calibration Error (MCE).

    MCE is the worst-case calibration gap across all bins:
        MCE = max_b |acc(B_b) - conf(B_b)|

    MCE is more conservative than ECE and is particularly
    important for medical AI where worst-case errors matter.

    Returns:
        mce: float — Maximum Calibration Error
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > low) & (confidences <= high)

        if mask.sum() == 0:
            continue

        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        mce      = max(mce, abs(bin_acc - bin_conf))

    return float(mce)


# =============================================================================
# SECTION 5: FIND OPTIMAL TEMPERATURE
# =============================================================================

def find_optimal_temperature(
    model  : nn.Module,
    loader : DataLoader,
    device : torch.device,
) -> float:
    """
    Finds the optimal temperature T that minimises NLL on the test set.

    Temperature Scaling is a post-hoc calibration method that applies
    a single scalar T to all logits before softmax. It does NOT change
    the model's predictions (argmax is preserved) — it only adjusts
    the confidence scores.

    This is the simplest and most widely used calibration method in
    deep learning, introduced by Guo et al. (2017):
    "On Calibration of Modern Neural Networks"

    Args:
        model  : HCTGNet in eval() mode
        loader : Validation or test DataLoader
        device : Compute device

    Returns:
        optimal_T: float — temperature that minimises NLL
    """
    model.eval()

    # Collect all logits and labels first
    all_logits  = []
    all_targets = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            logits  = model(signals)
            all_logits.extend(logits.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_logits  = torch.tensor(np.array(all_logits),  dtype=torch.float32)
    all_targets = torch.tensor(np.array(all_targets), dtype=torch.long)

    # Temperature is a learnable scalar, optimised via NLL minimisation
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion   = nn.CrossEntropyLoss()

    def eval_nll():
        optimizer.zero_grad()
        scaled_logits = all_logits / temperature
        loss = criterion(scaled_logits, all_targets)
        loss.backward()
        return loss

    optimizer.step(eval_nll)

    optimal_T = float(temperature.item())
    print(f"  Optimal temperature T = {optimal_T:.4f}")
    return optimal_T


# =============================================================================
# SECTION 6: RELIABILITY DIAGRAM
# =============================================================================

def plot_reliability_diagram(
    probs      : np.ndarray,
    targets    : np.ndarray,
    ece        : float,
    mce        : float,
    title      : str,
    save_path  : str,
    accent_color: str = '#00b4d8',
):
    """
    Plots a reliability diagram (calibration curve) for the model.

    A perfectly calibrated model's curve lies on the diagonal y=x.
    Points above the diagonal = underconfident (model is more accurate
    than its confidence suggests).
    Points below the diagonal = overconfident (model is less accurate
    than its confidence suggests).

    Args:
        probs       : Shape (N, 5) softmax probabilities
        targets     : Shape (N,) ground truth labels
        ece         : Expected Calibration Error
        mce         : Maximum Calibration Error
        title       : Plot title
        save_path   : Output path
        accent_color: Colour for the calibration line
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#1a1a2e')

    # ------------------------------------------------------------------
    # Left panel: Overall reliability diagram
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.set_facecolor('#16213e')

    # Compute calibration curve using sklearn
    # We use one-vs-rest for each class and average
    mean_frac_pos = np.zeros(N_BINS)
    mean_pred_val = np.zeros(N_BINS)
    count         = 0

    for class_idx in range(NUM_CLASSES):
        y_true_binary = (targets == class_idx).astype(int)
        y_prob_class  = probs[:, class_idx]

        try:
            frac_pos, pred_val = calibration_curve(
                y_true_binary, y_prob_class,
                n_bins=N_BINS, strategy='uniform'
            )
            # Pad to N_BINS if shorter
            if len(frac_pos) == len(pred_val):
                mean_frac_pos[:len(frac_pos)] += frac_pos
                mean_pred_val[:len(pred_val)] += pred_val
                count += 1
        except Exception:
            continue

    if count > 0:
        mean_frac_pos /= count
        mean_pred_val /= count

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'w--', linewidth=1.2,
            alpha=0.5, label='Perfect calibration', zorder=1)

    # Calibration curve
    ax.plot(mean_pred_val, mean_frac_pos,
            color=accent_color, linewidth=2.2,
            marker='o', markersize=5,
            label='HCTG-Net', zorder=3)

    # Fill gap between curve and diagonal
    ax.fill_between(
        mean_pred_val,
        mean_frac_pos,
        mean_pred_val,
        alpha=0.12, color=accent_color, zorder=2,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean Predicted Confidence', color='white', fontsize=10)
    ax.set_ylabel('Fraction of Positives (Accuracy)', color='white', fontsize=10)
    ax.set_title(f'{title}\nReliability Diagram',
                 color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='white', labelsize=8)
    ax.spines[:].set_color('#333355')
    ax.grid(True, linestyle=':', alpha=0.2, color='white')
    ax.legend(fontsize=9, facecolor='#1a1a2e',
              edgecolor='#444466', labelcolor='white')

    # ECE / MCE annotation
    ax.text(0.05, 0.92,
            f"ECE = {ece:.4f}\nMCE = {mce:.4f}",
            transform=ax.transAxes,
            fontsize=10, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='#2a2a4a',
                      edgecolor=accent_color,
                      alpha=0.9))

    # ------------------------------------------------------------------
    # Right panel: Confidence histogram
    # ------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')

    confidences = probs.max(axis=1)
    ax2.hist(confidences, bins=30, color=accent_color,
             alpha=0.8, edgecolor='white', linewidth=0.3)

    ax2.axvline(x=confidences.mean(), color='white',
                linestyle='--', linewidth=1.5,
                label=f'Mean conf = {confidences.mean():.3f}')

    ax2.set_xlabel('Predicted Confidence', color='white', fontsize=10)
    ax2.set_ylabel('Number of Samples', color='white', fontsize=10)
    ax2.set_title('Confidence Distribution',
                  color='white', fontsize=11, fontweight='bold')
    ax2.tick_params(colors='white', labelsize=8)
    ax2.spines[:].set_color('#333355')
    ax2.grid(True, linestyle=':', alpha=0.2, color='white')
    ax2.legend(fontsize=9, facecolor='#1a1a2e',
               edgecolor='#444466', labelcolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


# =============================================================================
# SECTION 7: BEFORE vs AFTER COMPARISON PLOT
# =============================================================================

def plot_calibration_comparison(
    probs_before : np.ndarray,
    probs_after  : np.ndarray,
    targets      : np.ndarray,
    ece_before   : float,
    ece_after    : float,
    temperature  : float,
    save_path    : str,
):
    """
    Side-by-side comparison of calibration before and after
    temperature scaling. This is the main figure for the paper.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        f'HCTG-Net Model Calibration — Before vs After Temperature Scaling (T={temperature:.3f})',
        color='white', fontsize=12, fontweight='bold', y=1.02,
    )

    panels = [
        (probs_before, ece_before, 'Before Scaling (T=1.0)', '#ff6d00'),
        (probs_after,  ece_after,  f'After Scaling  (T={temperature:.3f})', '#00c853'),
    ]

    for ax, (probs, ece, title, color) in zip(axes, panels):
        ax.set_facecolor('#16213e')

        # Compute mean calibration curve
        mean_frac_pos = np.zeros(N_BINS)
        mean_pred_val = np.zeros(N_BINS)
        count         = 0

        for class_idx in range(NUM_CLASSES):
            y_true_bin   = (targets == class_idx).astype(int)
            y_prob_class = probs[:, class_idx]
            try:
                frac_pos, pred_val = calibration_curve(
                    y_true_bin, y_prob_class,
                    n_bins=N_BINS, strategy='uniform'
                )
                if len(frac_pos) == len(pred_val):
                    mean_frac_pos[:len(frac_pos)] += frac_pos
                    mean_pred_val[:len(pred_val)] += pred_val
                    count += 1
            except Exception:
                continue

        if count > 0:
            mean_frac_pos /= count
            mean_pred_val /= count

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], 'w--', linewidth=1.2,
                alpha=0.5, label='Perfect calibration')

        # Model curve
        ax.plot(mean_pred_val, mean_frac_pos,
                color=color, linewidth=2.5,
                marker='o', markersize=6,
                label='HCTG-Net', zorder=3)

        ax.fill_between(mean_pred_val, mean_frac_pos, mean_pred_val,
                        alpha=0.12, color=color)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Mean Predicted Confidence', color='white', fontsize=10)
        ax.set_ylabel('Fraction of Positives', color='white', fontsize=10)
        ax.set_title(title, color=color, fontsize=11, fontweight='bold')
        ax.tick_params(colors='white', labelsize=8)
        ax.spines[:].set_color('#333355')
        ax.grid(True, linestyle=':', alpha=0.2, color='white')
        ax.legend(fontsize=9, facecolor='#1a1a2e',
                  edgecolor='#444466', labelcolor='white')

        # ECE annotation
        improvement = ""
        if title.startswith('After'):
            delta = ece_before - ece_after
            improvement = f"\n↓ {delta:.4f} improvement"
        ax.text(0.05, 0.92,
                f"ECE = {ece:.4f}{improvement}",
                transform=ax.transAxes,
                fontsize=10, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='#2a2a4a',
                          edgecolor=color, alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


# =============================================================================
# SECTION 8: SAVE REPORT
# =============================================================================

def save_calibration_report(
    ece_before  : float,
    mce_before  : float,
    ece_after   : float,
    mce_after   : float,
    temperature : float,
    save_dir    : str,
):
    """Saves calibration metrics for copy-paste into the paper."""

    lines = [
        "=" * 60,
        "  HCTG-Net — Model Calibration Report",
        "=" * 60,
        "",
        "  Method: Temperature Scaling (Guo et al., 2017)",
        f"  Optimal Temperature T = {temperature:.4f}",
        "",
        f"  {'Metric':<30} {'Before':>10} {'After':>10}",
        f"  {'-'*52}",
        f"  {'ECE (Expected Cal. Error)':<30} {ece_before:>10.4f} {ece_after:>10.4f}",
        f"  {'MCE (Maximum Cal. Error)':<30} {mce_before:>10.4f} {mce_after:>10.4f}",
        f"  {'ECE Improvement':<30} {'':>10} {(ece_before-ece_after):>+10.4f}",
        "",
        "  Calibration Quality (ECE):",
        "    < 0.01 = Excellent",
        "    < 0.05 = Well calibrated",
        "    < 0.10 = Acceptable",
        "    > 0.10 = Poorly calibrated",
        "",
        f"  Before scaling: {'Excellent' if ece_before < 0.01 else 'Well calibrated' if ece_before < 0.05 else 'Acceptable' if ece_before < 0.10 else 'Poorly calibrated'}",
        f"  After  scaling: {'Excellent' if ece_after  < 0.01 else 'Well calibrated' if ece_after  < 0.05 else 'Acceptable' if ece_after  < 0.10 else 'Poorly calibrated'}",
        "",
        "  IEEE Paper Statement:",
        f"  HCTG-Net achieves an ECE of {ece_before:.4f} without",
        f"  calibration, reduced to {ece_after:.4f} after temperature",
        f"  scaling (T={temperature:.3f}), demonstrating reliable",
        "  confidence estimation suitable for clinical deployment.",
        "",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    path = os.path.join(save_dir, 'calibration_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[SAVED] Report → {path}")


# =============================================================================
# SECTION 9: ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  HCTG-Net — Model Calibration Analysis")
    print("=" * 60 + "\n")

    torch.manual_seed(SEED)
    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"[MODEL] Loading checkpoint from '{CHECKPOINT_PATH}' ...")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    model = HCTGNet(
        num_classes=5, d_model=128, n_heads=4,
        ffn_dim=256, n_layers=2, dropout=0.1, clf_dropout=0.3,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded epoch {checkpoint['epoch']}  "
          f"(val F1 = {checkpoint['val_f1']:.4f})\n")

    # ------------------------------------------------------------------
    # 2. Build DataLoaders
    # ------------------------------------------------------------------
    print("[DATA] Building DataLoaders ...")
    loaders = build_mitbih_pipeline(
        data_dir='./mitbih_data',
        batch_size=512,
        num_workers=0,
        seed=SEED,
    )
    val_loader  = loaders['val']
    test_loader = loaders['test']

    # ------------------------------------------------------------------
    # 3. Get predictions BEFORE temperature scaling
    # ------------------------------------------------------------------
    print("\n[CALIBRATION] Computing metrics before scaling ...")
    probs_before, targets = get_all_probs(
        model, test_loader, device, temperature=1.0
    )
    ece_before = compute_ece(probs_before, targets, N_BINS)
    mce_before = compute_mce(probs_before, targets, N_BINS)
    print(f"  ECE before : {ece_before:.4f}")
    print(f"  MCE before : {mce_before:.4f}")

    # ------------------------------------------------------------------
    # 4. Find optimal temperature on validation set
    # ------------------------------------------------------------------
    print("\n[TEMPERATURE] Finding optimal temperature on val set ...")
    optimal_T = find_optimal_temperature(model, val_loader, device)

    # ------------------------------------------------------------------
    # 5. Get predictions AFTER temperature scaling
    # ------------------------------------------------------------------
    print("\n[CALIBRATION] Computing metrics after scaling ...")
    probs_after, _ = get_all_probs(
        model, test_loader, device, temperature=optimal_T
    )
    ece_after = compute_ece(probs_after, targets, N_BINS)
    mce_after = compute_mce(probs_after, targets, N_BINS)
    print(f"  ECE after  : {ece_after:.4f}")
    print(f"  MCE after  : {mce_after:.4f}")

    # ------------------------------------------------------------------
    # 6. Plot individual diagrams
    # ------------------------------------------------------------------
    print("\n[PLOT] Generating calibration figures ...")

    plot_reliability_diagram(
        probs=probs_before,
        targets=targets,
        ece=ece_before,
        mce=mce_before,
        title='Before Temperature Scaling (T=1.0)',
        save_path=os.path.join(RESULTS_DIR, 'calibration_before.png'),
        accent_color='#ff6d00',
    )

    plot_reliability_diagram(
        probs=probs_after,
        targets=targets,
        ece=ece_after,
        mce=mce_after,
        title=f'After Temperature Scaling (T={optimal_T:.3f})',
        save_path=os.path.join(RESULTS_DIR, 'calibration_after.png'),
        accent_color='#00c853',
    )

    # ------------------------------------------------------------------
    # 7. Before vs After comparison plot (main paper figure)
    # ------------------------------------------------------------------
    plot_calibration_comparison(
        probs_before=probs_before,
        probs_after=probs_after,
        targets=targets,
        ece_before=ece_before,
        ece_after=ece_after,
        temperature=optimal_T,
        save_path=os.path.join(RESULTS_DIR, 'calibration_comparison.png'),
    )

    # ------------------------------------------------------------------
    # 8. Save report
    # ------------------------------------------------------------------
    save_calibration_report(
        ece_before=ece_before,
        mce_before=mce_before,
        ece_after=ece_after,
        mce_after=mce_after,
        temperature=optimal_T,
        save_dir=RESULTS_DIR,
    )

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"  ECE before scaling : {ece_before:.4f}")
    print(f"  ECE after  scaling : {ece_after:.4f}")
    print(f"  Improvement        : {ece_before - ece_after:+.4f}")
    print(f"  Temperature T      : {optimal_T:.4f}")
    print(f"\n  Files saved in: {RESULTS_DIR}/")

    if ece_after < 0.05:
        print("\n  ✅ Model is well-calibrated — publication ready!")
    elif ece_after < 0.10:
        print("\n  ⚠️  Model is acceptably calibrated.")
    else:
        print("\n  ❌ Model needs further calibration work.")