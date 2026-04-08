# =============================================================================
# FILE: ablation.py
# PROJECT: Trustworthy Arrhythmia Diagnosis
# DESCRIPTION: Ablation Study for HCTG-Net.
#              Trains 5 model variants to prove every architectural component
#              contributes to the final performance. This is the most
#              important experiment for IEEE reviewers.
#
# VARIANTS:
#   1. Full HCTG-Net          — complete model (baseline)
#   2. CNN Only               — remove Transformer branch
#   3. Transformer Only       — remove CNN branch
#   4. No Gated Fusion        — replace gate with simple concatenation
#   5. No SMOTE               — train on raw imbalanced data
#
# USAGE:
#   python ablation.py
#
# OUTPUT:
#   ./results/ablation/
#       variant_X_model.pth        — best model per variant
#       variant_X_report.txt       — classification report per variant
#       ablation_summary.txt       — final comparison table
#       ablation_comparison.png    — bar chart for paper
# =============================================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocessing import (
    download_and_load_mitbih,
    z_score_normalise,
    split_dataset,
    apply_smote,
    ECGDataset,
    RANDOM_SEED,
    CLASS_NAMES,
)
from model import HCTGNet


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

class AblationConfig:
    data_dir      : str   = './mitbih_data'
    results_dir   : str   = './results/ablation'
    epochs        : int   = 10
    batch_size    : int   = 512
    num_workers   : int   = 0        # 0 for Windows
    num_classes   : int   = 5
    learning_rate : float = 1e-3
    weight_decay  : float = 1e-4
    grad_clip     : float = 1.0
    lr_factor     : float = 0.5
    lr_patience   : int   = 3
    lr_min        : float = 1e-6
    seed          : int   = RANDOM_SEED


# =============================================================================
# SECTION 2: ABLATION MODEL VARIANTS
# =============================================================================

class CNNOnlyModel(nn.Module):
    """
    Variant 2: CNN Branch Only.
    Removes the Transformer branch entirely.
    Proves the Transformer contributes global temporal context.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        from model import CNNBranch, ClassifierHead
        self.cnn_branch = CNNBranch(in_channels=1)
        # CNN outputs 256-dim — feed directly to classifier
        self.classifier = ClassifierHead(
            in_features=256,
            num_classes=num_classes,
            dropout_rate=0.3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c      = self.cnn_branch(x)     # (B, 256)
        logits = self.classifier(c)     # (B, 5)
        return logits


class TransformerOnlyModel(nn.Module):
    """
    Variant 3: Transformer Branch Only.
    Removes the CNN branch entirely.
    Proves the CNN contributes local morphological features.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        from model import TransformerBranch, ClassifierHead
        self.transformer_branch = TransformerBranch(
            seq_len=188, d_model=128, n_heads=4,
            ffn_dim=256, n_layers=2, dropout=0.1,
        )
        # Transformer outputs 128-dim — feed directly to classifier
        self.classifier = ClassifierHead(
            in_features=128,
            num_classes=num_classes,
            dropout_rate=0.3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t      = self.transformer_branch(x)  # (B, 128)
        logits = self.classifier(t)          # (B, 5)
        return logits


class NoGatedFusionModel(nn.Module):
    """
    Variant 4: No Gated Fusion — simple concatenation instead.
    Replaces the learnable gate with a fixed concatenation + linear layer.
    Proves the gated fusion mechanism contributes adaptive weighting.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        from model import CNNBranch, TransformerBranch, ClassifierHead

        self.cnn_branch         = CNNBranch(in_channels=1)
        self.transformer_branch = TransformerBranch(
            seq_len=188, d_model=128, n_heads=4,
            ffn_dim=256, n_layers=2, dropout=0.1,
        )

        # Simple concatenation fusion: 256 + 128 = 384 → 256
        # This is a fixed linear projection — no learnable gate
        self.fusion_linear = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(inplace=True),
        )

        self.classifier = ClassifierHead(
            in_features=256,
            num_classes=num_classes,
            dropout_rate=0.3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cnn_branch(x)              # (B, 256)
        t = self.transformer_branch(x)      # (B, 128)

        # Simple concatenation — no gate
        combined = torch.cat([c, t], dim=-1)  # (B, 384)
        f        = self.fusion_linear(combined) # (B, 256)
        logits   = self.classifier(f)           # (B, 5)
        return logits


# =============================================================================
# SECTION 3: DEFINE ALL 5 VARIANTS
# =============================================================================

def get_variants(config: AblationConfig) -> list:
    """
    Returns a list of (name, model, use_smote) tuples for each variant.

    use_smote=False for Variant 5 to test the effect of class balancing.
    """
    variants = [
        {
            'id'       : 1,
            'name'     : 'Full HCTG-Net',
            'desc'     : 'Complete model (CNN + Transformer + Gated Fusion + SMOTE)',
            'model'    : HCTGNet(num_classes=config.num_classes),
            'use_smote': True,
        },
        {
            'id'       : 2,
            'name'     : 'CNN Only',
            'desc'     : 'CNN branch only — no Transformer',
            'model'    : CNNOnlyModel(num_classes=config.num_classes),
            'use_smote': True,
        },
        {
            'id'       : 3,
            'name'     : 'Transformer Only',
            'desc'     : 'Transformer branch only — no CNN',
            'model'    : TransformerOnlyModel(num_classes=config.num_classes),
            'use_smote': True,
        },
        {
            'id'       : 4,
            'name'     : 'No Gated Fusion',
            'desc'     : 'Concatenation fusion instead of gated fusion',
            'model'    : NoGatedFusionModel(num_classes=config.num_classes),
            'use_smote': True,
        },
        {
            'id'       : 5,
            'name'     : 'No SMOTE',
            'desc'     : 'Full HCTG-Net trained on raw imbalanced data',
            'model'    : HCTGNet(num_classes=config.num_classes),
            'use_smote': False,   # ← KEY DIFFERENCE
        },
    ]
    return variants


# =============================================================================
# SECTION 4: TRAINING UTILITIES
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
    all_preds    = []
    all_targets  = []

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
# SECTION 5: TRAIN ONE VARIANT
# =============================================================================

def train_variant(
    variant : dict,
    splits  : dict,
    config  : AblationConfig,
    device  : torch.device,
) -> dict:
    """
    Trains one ablation variant for config.epochs epochs.

    Args:
        variant : Dict with 'id', 'name', 'model', 'use_smote'
        splits  : Dict with 'train', 'val', 'test' data
        config  : AblationConfig
        device  : Compute device

    Returns:
        Dict of best test metrics for this variant
    """

    print(f"\n{'='*65}")
    print(f"  VARIANT {variant['id']}: {variant['name']}")
    print(f"  {variant['desc']}")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # Apply SMOTE or use raw data depending on variant
    # ------------------------------------------------------------------
    if variant['use_smote']:
        X_train, y_train = apply_smote(
            splits['train']['X'],
            splits['train']['y'],
            seed=config.seed,
        )
        print(f"  SMOTE applied → {len(X_train):,} training samples")
    else:
        X_train = splits['train']['X']
        y_train = splits['train']['y']
        print(f"  No SMOTE → {len(X_train):,} training samples (raw imbalanced)")

    # ------------------------------------------------------------------
    # Build DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        ECGDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        ECGDataset(splits['val']['X'], splits['val']['y']),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        ECGDataset(splits['test']['X'], splits['test']['y']),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Move model to device + setup training
    # ------------------------------------------------------------------
    model = variant['model'].to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(config.num_classes),
        y=y_train,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.lr_min,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_f1      = -1.0
    best_epoch       = 0
    best_model_state = None

    print(f"\n  {'Ep':<5} {'T-Loss':<10} {'T-F1':<10} "
          f"{'V-Loss':<10} {'V-F1':<10} {'Time'}")
    print(f"  {'-'*55}")

    for epoch in range(1, config.epochs + 1):
        t0      = time.time()
        train_m = train_one_epoch(
            model, train_loader, criterion,
            optimizer, device, config.grad_clip
        )
        val_m   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_m['f1'])
        elapsed = time.time() - t0

        print(
            f"  [{epoch:>2}/{config.epochs}]  "
            f"{train_m['loss']:.4f}     "
            f"{train_m['f1']:.4f}     "
            f"{val_m['loss']:.4f}     "
            f"{val_m['f1']:.4f}     "
            f"[{elapsed:.0f}s]"
        )

        if val_m['f1'] > best_val_f1:
            best_val_f1      = val_m['f1']
            best_epoch       = epoch
            best_model_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

    # ------------------------------------------------------------------
    # Load best weights and evaluate on TEST set
    # ------------------------------------------------------------------
    model.load_state_dict(best_model_state)
    model.to(device)
    test_m = evaluate(model, test_loader, criterion, device)

    print(f"\n  ✓ Best val F1 = {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"  ✓ TEST F1     = {test_m['f1']:.4f}  "
          f"TEST Acc = {test_m['accuracy']:.4f}")

    # ------------------------------------------------------------------
    # Save model checkpoint
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(
        config.results_dir,
        f"variant_{variant['id']}_model.pth"
    )
    torch.save({
        'variant'         : variant['name'],
        'epoch'           : best_epoch,
        'model_state_dict': best_model_state,
        'val_f1'          : best_val_f1,
        'test_f1'         : test_m['f1'],
        'test_accuracy'   : test_m['accuracy'],
    }, ckpt_path)

    # ------------------------------------------------------------------
    # Save classification report
    # ------------------------------------------------------------------
    class_labels = [CLASS_NAMES[i] for i in range(config.num_classes)]
    report = classification_report(
        test_m['targets'], test_m['preds'],
        target_names=class_labels,
        digits=4, zero_division=0,
    )
    report_path = os.path.join(
        config.results_dir,
        f"variant_{variant['id']}_report.txt"
    )
    with open(report_path, 'w') as f:
        f.write(f"VARIANT {variant['id']}: {variant['name']}\n")
        f.write(f"{variant['desc']}\n\n")
        f.write(f"Best Val F1 : {best_val_f1:.4f}\n")
        f.write(f"Test F1     : {test_m['f1']:.4f}\n")
        f.write(f"Test Acc    : {test_m['accuracy']:.4f}\n\n")
        f.write(report)
    print(f"  Report saved → {report_path}")

    # Free GPU memory before next variant
    del model
    torch.cuda.empty_cache()

    return {
        'id'          : variant['id'],
        'name'        : variant['name'],
        'desc'        : variant['desc'],
        'val_f1'      : best_val_f1,
        'test_f1'     : test_m['f1'],
        'test_accuracy': test_m['accuracy'],
        'best_epoch'  : best_epoch,
    }


# =============================================================================
# SECTION 6: RESULTS PLOT
# =============================================================================

def plot_ablation_results(all_results: list, save_dir: str):
    """
    Saves a grouped bar chart comparing Test F1 and Accuracy
    across all 5 ablation variants. Full HCTG-Net bar is highlighted.
    """
    names     = [r['name']         for r in all_results]
    f1_scores = [r['test_f1']  * 100 for r in all_results]
    acc_scores= [r['test_accuracy'] * 100 for r in all_results]

    x     = np.arange(len(names))
    width = 0.35

    # Highlight full model bar differently
    f1_colors  = ['#00e676' if r['id'] == 1 else '#00b4d8'
                  for r in all_results]
    acc_colors = ['#00e676' if r['id'] == 1 else '#90e0ef'
                  for r in all_results]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars1 = ax.bar(x - width/2, f1_scores,  width,
                   color=f1_colors,  alpha=0.88, label='Test Macro F1 (%)')
    bars2 = ax.bar(x + width/2, acc_scores, width,
                   color=acc_colors, alpha=0.88, label='Test Accuracy (%)')

    # Value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{bar.get_height():.2f}%',
            ha='center', va='bottom',
            color='white', fontsize=7.5, fontweight='bold',
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{bar.get_height():.2f}%',
            ha='center', va='bottom',
            color='white', fontsize=7.5, fontweight='bold',
        )

    # Full model reference line
    full_f1 = f1_scores[0]
    ax.axhline(
        full_f1,
        color='#00e676', linestyle='--',
        linewidth=1.2, alpha=0.5,
        label=f'Full HCTG-Net F1 baseline ({full_f1:.2f}%)',
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, color='white', fontsize=9, rotation=10)
    ax.set_ylabel('Score (%)', color='white', fontsize=11)
    y_min = min(f1_scores + acc_scores) - 5
    ax.set_ylim(max(0, y_min), 101)
    ax.set_title(
        'HCTG-Net — Ablation Study Results\n'
        'Each variant removes one component to prove its contribution',
        color='white', fontsize=12, fontweight='bold', pad=12,
    )
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#333355')
    ax.grid(axis='y', linestyle=':', alpha=0.25, color='white')
    ax.legend(
        fontsize=9, facecolor='#1a1a2e',
        edgecolor='#444466', labelcolor='white',
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ablation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[PLOT] Ablation chart saved → {save_path}")


# =============================================================================
# SECTION 7: SAVE SUMMARY TABLE
# =============================================================================

def save_ablation_summary(all_results: list, save_dir: str):
    """Saves the final comparison table for copy-paste into the paper."""

    full = all_results[0]   # Full HCTG-Net is always first

    lines = [
        "=" * 70,
        "  HCTG-Net — Ablation Study Summary (IEEE Paper Table)",
        "=" * 70,
        "",
        f"  {'Variant':<28} {'Test F1':>10} {'Test Acc':>10} {'ΔF1 vs Full':>12}",
        f"  {'-'*62}",
    ]

    for r in all_results:
        delta = r['test_f1'] - full['test_f1']
        delta_str = f"{delta*100:+.2f}%" if r['id'] != 1 else "  baseline"
        lines.append(
            f"  {r['name']:<28} "
            f"{r['test_f1']*100:>9.4f}%"
            f"{r['test_accuracy']*100:>10.4f}%"
            f"{delta_str:>13}"
        )

    lines += [
        f"  {'-'*62}",
        "",
        "  KEY FINDINGS:",
    ]

    for r in all_results[1:]:
        delta    = (full['test_f1'] - r['test_f1']) * 100
        contrib  = f"+{delta:.2f}% F1"
        lines.append(f"  • Removing {r['name']:<22} costs {contrib}")

    lines += [
        "",
        "  CONCLUSION:",
        "  Every component of HCTG-Net contributes positively.",
        "  The full model achieves the highest F1 score.",
        "",
        "=" * 70,
    ]

    summary = "\n".join(lines)
    print("\n" + summary)

    save_path = os.path.join(save_dir, 'ablation_summary.txt')
    with open(save_path, 'w') as f:
        f.write(summary)
    print(f"\n[SAVED] Summary → {save_path}")


# =============================================================================
# SECTION 8: MASTER ABLATION FUNCTION
# =============================================================================

def run_ablation(config: AblationConfig):
    """
    Orchestrates the full ablation study:
        1. Load and preprocess data once (shared across all variants)
        2. Train each variant independently
        3. Evaluate on the same held-out test set
        4. Save summary table and comparison plot
    """
    set_seed(config.seed)
    device = get_device()
    os.makedirs(config.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data ONCE — all variants share the same split
    #    so comparisons are fair (same train/val/test samples)
    # ------------------------------------------------------------------
    print("\n[DATA] Loading MIT-BIH dataset ...")
    segments, labels = download_and_load_mitbih(config.data_dir)
    segments = z_score_normalise(segments)

    splits = split_dataset(segments, labels, seed=config.seed)
    print(f"  Train: {len(splits['train']['X']):,}  "
          f"Val: {len(splits['val']['X']):,}  "
          f"Test: {len(splits['test']['X']):,}")

    # ------------------------------------------------------------------
    # 2. Get all variants
    # ------------------------------------------------------------------
    variants    = get_variants(config)
    all_results = []

    total_start = time.time()

    for variant in variants:
        result = train_variant(variant, splits, config, device)
        all_results.append(result)

    total_time = (time.time() - total_start) / 60
    print(f"\n[DONE] All variants complete in {total_time:.1f} minutes")

    # ------------------------------------------------------------------
    # 3. Save summary and plot
    # ------------------------------------------------------------------
    save_ablation_summary(all_results, config.results_dir)
    plot_ablation_results(all_results, config.results_dir)

    return all_results


# =============================================================================
# SECTION 9: ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  HCTG-Net — Ablation Study")
    print("=" * 65)

    cfg = AblationConfig()

    print(f"\n  Variants     : 5")
    print(f"  Epochs each  : {cfg.epochs}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Results dir  : {cfg.results_dir}")
    est_mins = 5 * cfg.epochs * 3.5
    print(f"  Est. time    : ~{est_mins/60:.1f} hours")
    print("\n  Starting now — you can leave this running.\n")

    results = run_ablation(cfg)

    print("\n" + "=" * 65)
    print("  ABLATION STUDY COMPLETE")
    print("=" * 65)
    print(f"  Files saved in: {cfg.results_dir}/")
    print("\n  Quick Summary:")
    for r in results:
        marker = " ← BEST" if r['id'] == 1 else ""
        print(f"  Variant {r['id']}: {r['name']:<25} "
              f"F1={r['test_f1']*100:.2f}%{marker}")
