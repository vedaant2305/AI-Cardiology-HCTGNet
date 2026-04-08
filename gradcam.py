# =============================================================================
# FILE: gradcam.py
# PROJECT: Trustworthy Arrhythmia Diagnosis
# DESCRIPTION: GradCAM explainability visualisation for HCTG-Net.
#              Produces publication-quality figures showing which parts
#              of the ECG waveform the model focuses on for each class.
#
#              GradCAM is superior to vanilla saliency maps because:
#              - It targets a specific layer (last residual block)
#              - It uses gradient-weighted feature map activations
#              - It is more robust to noise in gradients
#              - It is the standard method cited in medical AI papers
#
# CLASSES VISUALISED:
#              N (Normal), V (PVC), S (Supraventricular), F (Fusion)
#
# USAGE:
#              python gradcam.py
#
# OUTPUT:
#              ./results/gradcam/
#                  gradcam_class_N.png
#                  gradcam_class_V.png
#                  gradcam_class_S.png
#                  gradcam_class_F.png
#                  gradcam_all_classes.png  <- 4-panel summary figure
# =============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from preprocessing import build_mitbih_pipeline, CLASS_NAMES
from model import HCTGNet


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

CHECKPOINT_PATH  = 'best_hctg_net.pth'
RESULTS_DIR      = './results/gradcam'
SAMPLING_RATE_HZ = 125
SEED             = 42

# Classes to visualise: (class_idx, display_name, accent_colour)
TARGET_CLASSES = [
    (0, 'Normal (N)',              '#00c853'),
    (2, 'Ventricular PVC (V)',     '#ff6d00'),
    (1, 'Supraventricular (S)',    '#ffd600'),
    (3, 'Fusion Beat (F)',         '#aa00ff'),
]


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
# SECTION 3: GRADCAM IMPLEMENTATION FOR 1D SIGNALS
# =============================================================================

class GradCAM1D:
    """
    GradCAM implementation adapted for 1D convolutional networks.

    Standard GradCAM (Selvaraju et al., 2017) was designed for 2D image
    CNNs. This implementation adapts it for 1D ECG signals by:
        - Hooking into the last residual block's output
        - Computing gradient-weighted channel activations along the
          time dimension instead of spatial dimensions
        - Upsampling the 1D CAM back to the original 188-sample length

    Algorithm:
        1. Forward pass → get feature maps A^k (C, L') from target layer
        2. Backward pass from class score → get gradients dY^c/dA^k
        3. Global average pool gradients: alpha^k = (1/L') * sum(dY^c/dA^k)
        4. Weighted combination: L^c = ReLU(sum_k(alpha^k * A^k))
        5. Upsample L^c from L' to 188 samples
        6. Normalise to [0, 1]

    Args:
        model        : HCTGNet in eval() mode
        target_layer : nn.Module — the layer to hook (last res block conv)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        # Register forward hook — captures feature maps during forward pass
        self._fwd_hook = target_layer.register_forward_hook(
            self._save_activations
        )

        # Register backward hook — captures gradients during backward pass
        self._bwd_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        """Stores feature maps from the forward pass."""
        self.activations = output.detach()   # (B, C, L')

    def _save_gradients(self, module, grad_input, grad_output):
        """Stores gradients from the backward pass."""
        self.gradients = grad_output[0].detach()  # (B, C, L')

    def generate(
        self,
        input_tensor  : torch.Tensor,
        target_class  : int,
        original_length: int = 188,
    ) -> np.ndarray:
        """
        Generates the GradCAM heatmap for a single input.

        Args:
            input_tensor    : Shape (1, 1, 188) — single ECG beat
            target_class    : Class index to explain
            original_length : Length to upsample CAM back to (188)

        Returns:
            cam: Shape (188,) — normalised GradCAM heatmap [0, 1]
        """
        self.model.eval()

        # Forward pass — activations are captured by the hook
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits       = self.model(input_tensor)       # (1, 5)

        # Zero all existing gradients
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        # Backward pass from the target class score
        # gradients are captured by the hook
        class_score = logits[0, target_class]
        class_score.backward()

        # Gradient-weighted activations
        # gradients  shape: (1, C, L')
        # activations shape: (1, C, L')
        gradients   = self.gradients[0]    # (C, L')
        activations = self.activations[0]  # (C, L')

        # Global average pool the gradients over the time dimension
        # alpha^k = (1/L') * sum_t(dY^c / dA^k_t)
        # Shape: (C,)
        weights = gradients.mean(dim=-1)

        # Weighted sum of activation maps
        # L^c = ReLU(sum_k(alpha^k * A^k))
        # Shape: (L',)
        cam = torch.zeros(activations.shape[-1],
                          device=activations.device)
        for k, w in enumerate(weights):
            cam += w * activations[k]

        # ReLU — only keep positive contributions
        cam = torch.relu(cam)

        # Upsample from L' back to original_length (188) using interpolation
        cam = cam.unsqueeze(0).unsqueeze(0)   # (1, 1, L')
        cam = torch.nn.functional.interpolate(
            cam,
            size=original_length,
            mode='linear',
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()     # (188,)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Call after use to free hook memory."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# =============================================================================
# SECTION 4: FIND CORRECTLY CLASSIFIED SAMPLE PER CLASS
# =============================================================================

def find_sample(
    model        : nn.Module,
    test_loader,
    target_class : int,
    device       : torch.device,
) -> tuple:
    """
    Finds the first correctly classified sample of target_class
    in the test set.

    Args:
        model        : HCTGNet in eval() mode
        test_loader  : Test DataLoader
        target_class : AAMI class index to find
        device       : Compute device

    Returns:
        sample (torch.Tensor): Shape (1, 1, 188) on CPU
        true_label (int)     : Ground truth class
    """
    model.eval()

    for signals, labels in test_loader:
        for i in range(signals.size(0)):
            if labels[i].item() != target_class:
                continue

            single = signals[i].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(single)
            pred = logits.argmax(dim=1).item()

            if pred == target_class:
                print(f"  Found Class-{target_class} "
                      f"('{CLASS_NAMES[target_class]}') sample")
                return signals[i].unsqueeze(0).cpu(), labels[i].item()

    raise RuntimeError(
        f"No correctly classified Class-{target_class} sample found."
    )


# =============================================================================
# SECTION 5: SINGLE CLASS GRADCAM PLOT
# =============================================================================

def plot_gradcam_single(
    waveform    : np.ndarray,
    cam         : np.ndarray,
    true_class  : int,
    pred_probs  : np.ndarray,
    class_name  : str,
    accent_color: str,
    save_path   : str,
):
    """
    Plots a single GradCAM visualisation with:
        - ECG waveform coloured by GradCAM activation intensity
        - Background heatmap gradient
        - R-peak marker
        - Confidence scores

    Args:
        waveform    : Shape (188,) normalised ECG
        cam         : Shape (188,) GradCAM heatmap [0,1]
        true_class  : Ground truth class index
        pred_probs  : Shape (5,) softmax probabilities
        class_name  : Display name e.g. 'Ventricular PVC (V)'
        accent_color: Hex colour for this class
        save_path   : Full output path
    """
    n_samples = len(waveform)
    time_ms   = np.arange(n_samples) * (1000.0 / SAMPLING_RATE_HZ)
    r_peak_ms = 90 * (1000.0 / SAMPLING_RATE_HZ)

    # Build coloured line collection
    cmap   = plt.get_cmap('RdYlBu_r')
    norm   = mcolors.Normalize(vmin=0.0, vmax=1.0)
    points = np.array([time_ms, waveform]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_cam= (cam[:-1] + cam[1:]) / 2.0
    lc     = LineCollection(segs, cmap=cmap, norm=norm,
                            linewidth=2.2, alpha=0.95, zorder=3)
    lc.set_array(seg_cam)

    fig = plt.figure(figsize=(13, 7))
    fig.patch.set_facecolor('#0f0f1a')
    gs  = fig.add_gridspec(2, 2,
                           height_ratios=[3, 1.2],
                           width_ratios=[30, 1],
                           hspace=0.08, wspace=0.04)

    ax_ecg  = fig.add_subplot(gs[0, 0])
    ax_cam  = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[:, 1])

    # --- ECG panel ---
    ax_ecg.set_facecolor('#0f0f1a')
    ax_ecg.add_collection(lc)

    # Background heatmap
    for i in range(n_samples - 1):
        ax_ecg.axvspan(
            time_ms[i], time_ms[i+1],
            alpha=cam[i] * 0.25,
            color=cmap(norm(cam[i])),
            zorder=1,
        )

    ax_ecg.axvline(x=r_peak_ms, color='white', linestyle='--',
                   linewidth=1.2, alpha=0.5, zorder=4, label='R-peak')

    y_pad = (waveform.max() - waveform.min()) * 0.18
    ax_ecg.set_xlim(time_ms[0], time_ms[-1])
    ax_ecg.set_ylim(waveform.min() - y_pad, waveform.max() + y_pad)
    ax_ecg.set_ylabel('Amplitude (z-score)', color='white', fontsize=10)
    ax_ecg.tick_params(colors='white', labelsize=8)
    ax_ecg.spines[:].set_color('#333355')
    ax_ecg.set_xticklabels([])
    ax_ecg.grid(True, linestyle=':', alpha=0.2, color='white')
    ax_ecg.legend(loc='upper right', fontsize=8,
                  facecolor='#1a1a2e', edgecolor='#444466',
                  labelcolor='white')

    # Confidence annotation
    conf = pred_probs[true_class] * 100
    ax_ecg.text(
        0.02, 0.95,
        f"Confidence: {conf:.1f}%",
        transform=ax_ecg.transAxes,
        fontsize=9, color=accent_color,
        fontweight='bold', va='top',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='#1a1a2e',
                  edgecolor=accent_color,
                  alpha=0.8),
    )

    # --- GradCAM panel ---
    ax_cam.set_facecolor('#0f0f1a')
    ax_cam.fill_between(time_ms, cam, alpha=0.75,
                        color='tomato', zorder=2)
    ax_cam.plot(time_ms, cam, color='white',
                linewidth=0.8, alpha=0.5, zorder=3)
    ax_cam.axvline(x=r_peak_ms, color='white', linestyle='--',
                   linewidth=1.2, alpha=0.5, zorder=4)
    ax_cam.set_xlim(time_ms[0], time_ms[-1])
    ax_cam.set_ylim(-0.05, 1.15)
    ax_cam.set_xlabel('Time (ms)', color='white', fontsize=10)
    ax_cam.set_ylabel('GradCAM', color='white', fontsize=10)
    ax_cam.tick_params(colors='white', labelsize=8)
    ax_cam.spines[:].set_color('#333355')
    ax_cam.grid(True, linestyle=':', alpha=0.2, color='white')

    # --- Colorbar ---
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Activation', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=7)
    cbar.outline.set_edgecolor('#555555')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # --- Title ---
    fig.text(0.5, 0.97,
             f"GradCAM Explainability — Class {class_name}",
             ha='center', va='top', fontsize=13,
             fontweight='bold', color='white')
    fig.text(0.5, 0.935,
             "Red/Yellow = high activation (model focused here)   "
             "Blue = low activation (model ignored here)",
             ha='center', va='top', fontsize=8.5,
             color='#aaaaaa', style='italic')

    plt.savefig(save_path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


# =============================================================================
# SECTION 6: 4-PANEL SUMMARY FIGURE
# =============================================================================

def plot_gradcam_summary(
    results    : list,
    save_path  : str,
):
    """
    Creates a 4-panel summary figure showing GradCAM for all classes
    side by side. This is the main figure for the IEEE paper.

    Each panel shows the ECG waveform coloured by GradCAM activation.

    Args:
        results   : List of dicts with waveform, cam, class info
        save_path : Output path
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle(
        'HCTG-Net GradCAM Explainability — All Classes\n'
        'Gradient-weighted Class Activation Maps (Last Residual Block)',
        fontsize=13, fontweight='bold', color='white', y=0.98,
    )

    cmap = plt.get_cmap('RdYlBu_r')
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for ax, r in zip(axes, results):
        ax.set_facecolor('#0f0f1a')

        waveform     = r['waveform']
        cam          = r['cam']
        accent_color = r['accent_color']
        class_name   = r['class_name']
        confidence   = r['confidence']

        n_samples = len(waveform)
        time_ms   = np.arange(n_samples) * (1000.0 / SAMPLING_RATE_HZ)
        r_peak_ms = 90 * (1000.0 / SAMPLING_RATE_HZ)

        # Background heatmap
        for i in range(n_samples - 1):
            ax.axvspan(
                time_ms[i], time_ms[i+1],
                alpha=cam[i] * 0.28,
                color=cmap(norm(cam[i])),
                zorder=1,
            )

        # Coloured waveform line
        points = np.array([time_ms, waveform]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_cam= (cam[:-1] + cam[1:]) / 2.0
        lc     = LineCollection(segs, cmap=cmap, norm=norm,
                                linewidth=1.8, alpha=0.95, zorder=3)
        lc.set_array(seg_cam)
        ax.add_collection(lc)

        # R-peak marker
        ax.axvline(x=r_peak_ms, color='white', linestyle='--',
                   linewidth=1.0, alpha=0.45, zorder=4)

        y_pad = (waveform.max() - waveform.min()) * 0.18
        ax.set_xlim(time_ms[0], time_ms[-1])
        ax.set_ylim(waveform.min() - y_pad, waveform.max() + y_pad)
        ax.tick_params(colors='white', labelsize=8)
        ax.spines[:].set_color('#333355')
        ax.grid(True, linestyle=':', alpha=0.18, color='white')

        # Class label on left
        ax.set_ylabel(
            f"Class {class_name}",
            color=accent_color,
            fontsize=10, fontweight='bold',
        )

        # Confidence on right
        ax.text(
            0.99, 0.88,
            f"{confidence:.1f}% confidence",
            transform=ax.transAxes,
            fontsize=8.5, color=accent_color,
            fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='#1a1a2e',
                      edgecolor=accent_color,
                      alpha=0.8),
        )

    axes[-1].set_xlabel('Time (ms)', color='white', fontsize=10)

    # Shared colorbar
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('GradCAM Activation', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=7)
    cbar.outline.set_edgecolor('#555555')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.subplots_adjust(hspace=0.12, right=0.90)
    plt.savefig(save_path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[PLOT] 4-panel summary saved → {save_path}")


# =============================================================================
# SECTION 7: ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  HCTG-Net — GradCAM Explainability")
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
    # 2. Build test DataLoader
    # ------------------------------------------------------------------
    print("[DATA] Building test DataLoader ...")
    loaders = build_mitbih_pipeline(
        data_dir='./mitbih_data',
        batch_size=256,
        num_workers=0,
        seed=SEED,
    )
    test_loader = loaders['test']

    # ------------------------------------------------------------------
    # 3. Set up GradCAM targeting the LAST residual block's second conv
    #
    # WHY this layer?
    #   - Early layers detect primitive features (edges, slopes)
    #   - The last residual block detects the highest-level patterns
    #     (QRS morphology, ST elevation, T-wave shape)
    #   - GradCAM on this layer shows the clinically meaningful features
    # ------------------------------------------------------------------
    target_layer = model.cnn_branch.res_block3.conv_path[3]
    # conv_path[3] is the second Conv1d in ResBlock3 (after BN+ReLU)
    # This is the highest-level feature extractor in the CNN branch

    gradcam = GradCAM1D(model=model, target_layer=target_layer)

    # ------------------------------------------------------------------
    # 4. Generate GradCAM for each target class
    # ------------------------------------------------------------------
    summary_results = []

    print("\n[GRADCAM] Generating visualisations ...\n")

    for class_idx, class_name, accent_color in TARGET_CLASSES:

        print(f"  Processing Class {CLASS_NAMES[class_idx]} "
              f"({class_name}) ...")

        # Find a correctly classified sample
        sample_input, true_label = find_sample(
            model, test_loader, class_idx, device
        )

        # Generate GradCAM
        input_on_device = sample_input.to(device)
        cam = gradcam.generate(
            input_tensor=input_on_device,
            target_class=class_idx,
            original_length=188,
        )

        # Get softmax probabilities for confidence display
        with torch.no_grad():
            logits     = model(input_on_device)
            pred_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        waveform   = sample_input.squeeze().numpy()   # (188,)
        confidence = pred_probs[class_idx] * 100

        print(f"    GradCAM range: [{cam.min():.3f}, {cam.max():.3f}]")
        print(f"    Confidence   : {confidence:.1f}%")

        # Save individual plot
        save_path = os.path.join(
            RESULTS_DIR,
            f"gradcam_class_{CLASS_NAMES[class_idx]}.png"
        )
        plot_gradcam_single(
            waveform=waveform,
            cam=cam,
            true_class=class_idx,
            pred_probs=pred_probs,
            class_name=class_name,
            accent_color=accent_color,
            save_path=save_path,
        )

        # Store for summary figure
        summary_results.append({
            'class_idx'   : class_idx,
            'class_name'  : class_name,
            'accent_color': accent_color,
            'waveform'    : waveform,
            'cam'         : cam,
            'confidence'  : confidence,
        })

        print()

    # ------------------------------------------------------------------
    # 5. Free hooks
    # ------------------------------------------------------------------
    gradcam.remove_hooks()

    # ------------------------------------------------------------------
    # 6. Generate 4-panel summary figure
    # ------------------------------------------------------------------
    summary_path = os.path.join(RESULTS_DIR, 'gradcam_all_classes.png')
    plot_gradcam_summary(summary_results, summary_path)

    # ------------------------------------------------------------------
    # 7. Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  GradCAM Complete! Files saved:")
    print("=" * 60)
    for r in summary_results:
        print(f"  gradcam_class_{CLASS_NAMES[r['class_idx']]}.png  "
              f"— {r['class_name']}  ({r['confidence']:.1f}% conf)")
    print(f"  gradcam_all_classes.png  — 4-panel summary for paper")
    print(f"\n  All saved in: {RESULTS_DIR}/")
    print("=" * 60)