
# =============================================================================
# FILE: app.py
# PROJECT: Trustworthy Arrhythmia Diagnosis — Clinical Dashboard
# DESCRIPTION: Streamlit web application that loads the trained HCTG-Net
#              model and provides a clean, clinician-facing interface for
#              single-beat ECG arrhythmia classification.
#
#              Workflow:
#                1. Clinician uploads a 188-point ECG beat as a CSV file.
#                2. App validates, normalises, and visualises the waveform.
#                3. HCTG-Net classifies the beat into one of 5 AAMI classes.
#                4. Diagnosis and per-class confidence scores are displayed.
#
# USAGE:
#   pip install streamlit torch numpy matplotlib
#   streamlit run app.py
#
# ASSUMES:
#   - best_hctg_net.pth  is in the working directory
#   - model.py           is in the working directory (defines HCTGNet)
# =============================================================================

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import streamlit as st

# Local module — must be in the same directory as app.py
from model import HCTGNet


# =============================================================================
# SECTION 1: CONSTANTS
# =============================================================================

# Path to the saved best-model checkpoint (relative to working directory)
CHECKPOINT_PATH = 'best_hctg_net.pth'

# Expected number of time-steps in every uploaded ECG beat
EXPECTED_SAMPLES = 188

# Post-downsampling rate used during preprocessing (125 Hz → 8 ms / sample)
SAMPLING_RATE_HZ = 125

# AAMI class metadata: index → (short label, full clinical name, accent colour)
# Colours are chosen to match standard clinical risk conventions:
#   green  = benign / normal
#   yellow = caution / monitor
#   orange = elevated risk
#   red    = high risk / urgent
AAMI_CLASSES = {
    0: {
        'label'      : 'N',
        'full_name'  : 'Normal / Bundle Branch Block',
        'description': 'Normal beat or bundle branch block. No immediate action required.',
        'risk'       : 'Low Risk',
        'color'      : '#00c853',   # green
        'bg_color'   : '#e8f5e9',
    },
    1: {
        'label'      : 'S',
        'full_name'  : 'Supraventricular Ectopic',
        'description': 'Atrial or junctional premature beat. Monitor for frequency and symptoms.',
        'risk'       : 'Moderate Risk',
        'color'      : '#ffd600',   # yellow
        'bg_color'   : '#fffde7',
    },
    2: {
        'label'      : 'V',
        'full_name'  : 'Ventricular Ectopic (PVC)',
        'description': 'Premature ventricular contraction. Clinician review recommended.',
        'risk'       : 'Elevated Risk',
        'color'      : '#ff6d00',   # orange
        'bg_color'   : '#fff3e0',
    },
    3: {
        'label'      : 'F',
        'full_name'  : 'Fusion Beat',
        'description': 'Fusion of normal and ventricular depolarisation. Further evaluation advised.',
        'risk'       : 'Elevated Risk',
        'color'      : '#aa00ff',   # purple
        'bg_color'   : '#f3e5f5',
    },
    4: {
        'label'      : 'Q',
        'full_name'  : 'Unknown / Unclassifiable',
        'description': 'Beat does not match a known morphology. Manual review required.',
        'risk'       : 'Indeterminate',
        'color'      : '#546e7a',   # grey-blue
        'bg_color'   : '#eceff1',
    },
}


# =============================================================================
# SECTION 2: CACHED MODEL LOADER
# =============================================================================

@st.cache_resource
def load_model() -> torch.nn.Module:
    """
    Loads the HCTG-Net architecture and restores the best saved weights.

    @st.cache_resource ensures this function runs EXACTLY ONCE per app
    session, regardless of how many times the user interacts with the UI.
    The loaded model object is cached in memory and reused on every
    subsequent call — avoiding the ~1-2 second reload penalty on each
    file upload.

    Returns:
        model: HCTGNet with best weights, on CPU, in eval() mode.
    """
    try:
        # Load the full checkpoint dict saved by train.py
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=torch.device('cpu'),   # always CPU for web deployment
        )

        # Reconstruct the model with the same hyperparameters used in training
        model = HCTGNet(num_classes=5)

        model.load_state_dict(checkpoint['model_state_dict'])

        # eval() is CRITICAL for deployment:
        #   - Disables Dropout (stochastic → deterministic predictions)
        #   - Freezes BatchNorm running statistics (uses training-time stats)
        # Without this, the same input can produce different outputs on
        # each call, which is clinically unacceptable.
        model.eval()

        return model

    except FileNotFoundError:
        # Surface a clear error in the UI rather than crashing silently
        st.error(
            f"❌ Checkpoint file `{CHECKPOINT_PATH}` not found. "
            "Please ensure the trained model weights are in the working directory."
        )
        st.stop()


# =============================================================================
# SECTION 3: PREPROCESSING HELPER
# =============================================================================

def preprocess_waveform(raw: np.ndarray) -> tuple:
    """
    Applies the same Z-score normalisation used during training so that the
    uploaded waveform is on the same scale the model was trained on.

    Normalising at inference time is mandatory — skipping it would shift
    the input distribution away from what the model learned, degrading
    classification accuracy (especially for amplitude-sensitive features
    like QRS peak height and T-wave polarity).

    Args:
        raw (np.ndarray): Shape (188,) — raw amplitude values from the CSV.

    Returns:
        normalised (np.ndarray): Shape (188,) — Z-score normalised waveform.
        tensor     (torch.Tensor): Shape (1, 1, 188) — model-ready input.
    """
    eps = 1e-6
    mu     = raw.mean()
    sigma  = raw.std()
    normalised = (raw - mu) / (sigma + eps)

    # Build the 3-D tensor expected by HCTG-Net: (Batch=1, Channel=1, Length=188)
    tensor = torch.tensor(normalised, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return normalised, tensor


# =============================================================================
# SECTION 4: INFERENCE
# =============================================================================

@torch.no_grad()
def run_inference(model: torch.nn.Module, tensor: torch.Tensor) -> tuple:
    """
    Runs the forward pass and converts raw logits to class probabilities.

    @torch.no_grad() wraps the entire function:
        - Disables gradient computation (the computational graph is not built).
        - Reduces memory consumption by ~50% vs. a normal forward pass.
        - Speeds up inference — essential for a responsive web UI.

    Args:
        model  : Loaded HCTGNet in eval() mode.
        tensor : Shape (1, 1, 188) — normalised, batched ECG beat.

    Returns:
        pred_class (int)       : Index of the predicted AAMI class (0–4).
        probs      (np.ndarray): Shape (5,) — softmax confidence scores [0, 1].
    """
    logits = model(tensor)                              # (1, 5) raw scores
    probs  = torch.softmax(logits, dim=1)               # (1, 5) probabilities
    probs  = probs.squeeze(0).numpy()                   # (5,)

    pred_class = int(probs.argmax())
    return pred_class, probs


# =============================================================================
# SECTION 5: ECG VISUALISATION
# =============================================================================

def plot_ecg(
    waveform : np.ndarray,
    pred_class: int,
    title_suffix: str = '',
) -> plt.Figure:
    """
    Renders a clean, clinical-style 1-D ECG waveform plot.

    Design choices:
        - Dark background with a light grid evokes a standard ECG monitor.
        - The waveform is coloured with the diagnosis accent colour so the
          visual result is immediately associated with the risk level.
        - A vertical dashed line marks the R-peak position (sample 90),
          which is the anatomical anchor of the 188-sample window.
        - Time axis is in milliseconds for clinical convention.

    Args:
        waveform    : Shape (188,) — normalised ECG values.
        pred_class  : Predicted AAMI class index (used for colour).
        title_suffix: Optional subtitle string (e.g. predicted label).

    Returns:
        fig: Matplotlib Figure object for st.pyplot().
    """
    class_info = AAMI_CLASSES[pred_class]
    line_color = class_info['color']

    n_samples = len(waveform)
    time_ms   = np.arange(n_samples) * (1000.0 / SAMPLING_RATE_HZ)
    r_peak_ms = 90 * (1000.0 / SAMPLING_RATE_HZ)   # R-peak at sample 90

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # --- Waveform line ---
    ax.plot(time_ms, waveform,
            color=line_color, linewidth=1.8, zorder=3, label='ECG Beat')

    # Subtle fill under the curve for visual depth
    ax.fill_between(time_ms, waveform, waveform.min() - 0.1,
                    color=line_color, alpha=0.08, zorder=2)

    # --- R-peak marker ---
    ax.axvline(x=r_peak_ms, color='#ffffff', linestyle='--',
               linewidth=1.0, alpha=0.45, zorder=4, label='R-peak')

    # --- Axis formatting ---
    y_pad = (waveform.max() - waveform.min()) * 0.20
    ax.set_xlim(time_ms[0], time_ms[-1])
    ax.set_ylim(waveform.min() - y_pad, waveform.max() + y_pad)
    ax.set_xlabel('Time (ms)', color='#cccccc', fontsize=10)
    ax.set_ylabel('Amplitude (z-score)', color='#cccccc', fontsize=10)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333355')
    ax.grid(True, linestyle=':', alpha=0.25, color='#aaaaaa')

    ax.legend(
        loc='upper right', fontsize=8,
        facecolor='#1a1a2e', edgecolor='#444466',
        labelcolor='#cccccc',
    )

    title_text = f"Uploaded ECG Beat  {title_suffix}"
    ax.set_title(title_text, color='white', fontsize=11, pad=10)

    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 6: CONFIDENCE BAR CHART
# =============================================================================

def plot_confidence_bars(probs: np.ndarray) -> plt.Figure:
    """
    Renders a horizontal bar chart of the per-class confidence scores.

    Each bar is coloured with the class accent colour so clinicians can
    immediately identify which classes the model was uncertain between.
    The predicted class bar has a white border highlight.

    Args:
        probs: Shape (5,) — softmax probabilities summing to 1.

    Returns:
        fig: Matplotlib Figure for st.pyplot().
    """
    pred_class = int(probs.argmax())
    class_labels = [
        f"{AAMI_CLASSES[i]['label']} — {AAMI_CLASSES[i]['full_name']}"
        for i in range(5)
    ]
    colors = [AAMI_CLASSES[i]['color'] for i in range(5)]
    pct    = probs * 100   # convert to percentages

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars = ax.barh(
        class_labels, pct,
        color=colors,
        edgecolor='none',
        height=0.55,
        zorder=3,
    )

    # Highlight the predicted class bar with a white border
    bars[pred_class].set_edgecolor('white')
    bars[pred_class].set_linewidth(1.8)

    # Value labels at the end of each bar
    for bar, val in zip(bars, pct):
        ax.text(
            min(val + 1.0, 97),
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%',
            va='center', ha='left',
            color='white', fontsize=8.5, fontweight='bold',
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel('Confidence (%)', color='#cccccc', fontsize=9)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.set_title('Per-Class Confidence Scores', color='white',
                 fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_color('#333355')
    ax.grid(axis='x', linestyle=':', alpha=0.2, color='#aaaaaa')
    ax.invert_yaxis()   # highest bar at the top

    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 7: STREAMLIT UI LAYOUT
# =============================================================================

def main():
    """
    Defines the full Streamlit page layout and application logic.

    Page flow:
        Header   → Sidebar instructions
        Uploader → Validation → ECG Plot → Diagnosis card → Confidence chart
        Footer   → Clinical disclaimer
    """

    # ------------------------------------------------------------------
    # Page config — must be the FIRST Streamlit call in the script
    # ------------------------------------------------------------------
    st.set_page_config(
        page_title="Arrhythmia Diagnosis — HCTG-Net",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ------------------------------------------------------------------
    # Inject minimal custom CSS for the diagnosis result card
    # ------------------------------------------------------------------
    st.markdown("""
    <style>
        .diagnosis-card {
            border-radius: 12px;
            padding: 24px 28px;
            margin: 16px 0;
            border-left: 6px solid;
        }
        .diagnosis-label {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            margin-bottom: 4px;
        }
        .diagnosis-name {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 8px;
            opacity: 0.85;
        }
        .diagnosis-desc {
            font-size: 0.92rem;
            opacity: 0.75;
            margin-bottom: 4px;
        }
        .risk-badge {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .metric-box {
            background: #1e1e2e;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------
    st.markdown("# 🫀 HCTG-Net Arrhythmia Diagnosis")
    st.markdown(
        "**Hybrid CNN–Transformer Network with Gated Fusion** | "
        "MIT-BIH Arrhythmia Database | 5-Class AAMI Classification"
    )
    st.divider()

    # ------------------------------------------------------------------
    # SIDEBAR — instructions and class legend
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 📋 How to Use")
        st.markdown("""
1. Export a **single 188-sample ECG heartbeat** as a `.csv` file.
2. The CSV must contain **one column** of floating-point amplitude values.
3. Upload the file using the uploader on the main panel.
4. The model will classify the beat and display the diagnosis.
        """)

        st.markdown("---")
        st.markdown("## 🏷️ AAMI Class Reference")

        for idx, info in AAMI_CLASSES.items():
            st.markdown(
                f"<span style='color:{info['color']}; font-weight:700;'>"
                f"Class {info['label']}</span> — {info['full_name']}",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("## ⚙️ Model Details")
        st.markdown("""
| Property | Value |
|---|---|
| Architecture | HCTG-Net |
| Dataset | MIT-BIH |
| Accuracy | 99.46% |
| Macro F1 | 97.11% |
| Input | 188 samples @ 125 Hz |
| Classes | 5 (AAMI standard) |
        """)

        st.markdown("---")
        st.caption(
            "⚠️ **Research prototype only.** "
            "Not approved for clinical diagnostic use. "
            "Always consult a qualified cardiologist."
        )

    # ------------------------------------------------------------------
    # LOAD MODEL (cached — only runs once per session)
    # ------------------------------------------------------------------
    model = load_model()

    # ------------------------------------------------------------------
    # FILE UPLOADER
    # ------------------------------------------------------------------
    st.markdown("### 📂 Upload ECG Beat (CSV)")
    st.markdown(
        "Upload a single-column CSV file containing exactly "
        f"**{EXPECTED_SAMPLES} amplitude samples** representing one heartbeat "
        "(90 pre-R-peak + 98 post-R-peak samples)."
    )

    uploaded_file = st.file_uploader(
        label="Choose a CSV file",
        type=["csv"],
        help=(
            f"The file must contain exactly {EXPECTED_SAMPLES} rows, "
            "one amplitude value per row, with no header."
        ),
    )

    # ------------------------------------------------------------------
    # MAIN ANALYSIS BLOCK — only runs when a file is uploaded
    # ------------------------------------------------------------------
    if uploaded_file is not None:

        st.divider()

        # --- Read and validate the uploaded CSV ---
        try:
            raw_data = np.loadtxt(
                io.StringIO(uploaded_file.read().decode('utf-8')),
                delimiter=',',
            )

            # Flatten in case the CSV has shape (188, 1) instead of (188,)
            raw_data = raw_data.flatten().astype(np.float32)

        except Exception as e:
            st.error(f"❌ Failed to read CSV file: `{e}`. "
                     "Please ensure the file is a valid numeric CSV.")
            st.stop()

        # --- Validate sample count ---
        if len(raw_data) != EXPECTED_SAMPLES:
            st.error(
                f"❌ Invalid input: expected **{EXPECTED_SAMPLES} samples**, "
                f"but received **{len(raw_data)}**. "
                "Please re-export the heartbeat with the correct window size."
            )
            st.stop()

        # --- Check for NaN / Inf values ---
        if not np.isfinite(raw_data).all():
            st.error(
                "❌ The uploaded file contains NaN or infinite values. "
                "Please check the signal quality and re-upload."
            )
            st.stop()

        st.success(
            f"✅ File accepted: `{uploaded_file.name}` — "
            f"{len(raw_data)} samples loaded successfully."
        )

        # --- Preprocess: Z-score normalise → build tensor ---
        normalised_wave, input_tensor = preprocess_waveform(raw_data)

        # --- Run inference ---
        with st.spinner("🔍 Analysing ECG beat..."):
            pred_class, probs = run_inference(model, input_tensor)

        class_info   = AAMI_CLASSES[pred_class]
        confidence   = float(probs[pred_class]) * 100

        # ==============================================================
        # LAYOUT: two columns — ECG plot (left) | Diagnosis card (right)
        # ==============================================================
        col_ecg, col_result = st.columns([1.6, 1.0], gap="large")

        # --- LEFT: ECG waveform plot ---
        with col_ecg:
            st.markdown("#### Uploaded ECG Waveform")
            fig_ecg = plot_ecg(
                waveform=normalised_wave,
                pred_class=pred_class,
                title_suffix=f"| Predicted: Class {class_info['label']}",
            )
            st.pyplot(fig_ecg, use_container_width=True)
            plt.close(fig_ecg)   # free memory immediately after rendering

            # Show raw signal statistics in a small expander
            with st.expander("📊 Signal Statistics"):
                s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                s_col1.metric("Min",    f"{normalised_wave.min():.3f}")
                s_col2.metric("Max",    f"{normalised_wave.max():.3f}")
                s_col3.metric("Mean",   f"{normalised_wave.mean():.3f}")
                s_col4.metric("Std",    f"{normalised_wave.std():.3f}")

        # --- RIGHT: Diagnosis result card ---
        with col_result:
            st.markdown("#### Diagnosis")

            # Coloured diagnosis card rendered via HTML/CSS
            st.markdown(f"""
            <div class="diagnosis-card"
                 style="background-color:{class_info['bg_color']};
                        border-color:{class_info['color']};">
                <div class="diagnosis-label" style="color:{class_info['color']};">
                    Class {class_info['label']}
                </div>
                <div class="diagnosis-name" style="color:#333;">
                    {class_info['full_name']}
                </div>
                <div class="diagnosis-desc" style="color:#555;">
                    {class_info['description']}
                </div>
                <br/>
                <span class="risk-badge"
                      style="background:{class_info['color']}; color:white;">
                    {class_info['risk']}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Confidence score metric
            st.markdown(f"""
            <div class="metric-box">
                <div style="color:#aaaaaa; font-size:0.85rem; margin-bottom:4px;">
                    MODEL CONFIDENCE
                </div>
                <div style="color:{class_info['color']};
                            font-size:2.8rem; font-weight:800;">
                    {confidence:.1f}%
                </div>
                <div style="color:#777; font-size:0.78rem;">
                    Softmax probability for Class {class_info['label']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ==============================================================
        # CONFIDENCE BAR CHART — full width below the two columns
        # ==============================================================
        st.markdown("#### Per-Class Confidence Breakdown")
        fig_bars = plot_confidence_bars(probs)
        st.pyplot(fig_bars, use_container_width=True)
        plt.close(fig_bars)

        # ==============================================================
        # RAW PROBABILITY TABLE — collapsible
        # ==============================================================
        with st.expander("🔢 Raw Probability Values"):
            import pandas as pd
            prob_df = pd.DataFrame({
                'Class'      : [AAMI_CLASSES[i]['label']     for i in range(5)],
                'Full Name'  : [AAMI_CLASSES[i]['full_name'] for i in range(5)],
                'Probability': [f"{probs[i]*100:.4f}%"       for i in range(5)],
            })
            st.dataframe(
                prob_df,
                use_container_width=True,
                hide_index=True,
            )

        # ==============================================================
        # CLINICAL DISCLAIMER — always visible after a result
        # ==============================================================
        st.divider()
        st.warning(
            "⚠️ **Clinical Disclaimer:** This output is generated by a "
            "research-grade deep learning model and has **not** been validated "
            "for clinical use. It must not be used as the sole basis for any "
            "medical decision. Always confirm findings with a qualified "
            "cardiologist and standard diagnostic equipment."
        )

    else:
        # Placeholder state — no file uploaded yet
        st.info(
            "👆 Upload a CSV file above to begin analysis. "
            "See the sidebar for formatting instructions."
        )

        # Show a sample waveform diagram so clinicians know what to expect
        st.markdown("---")
        st.markdown("#### Example: 188-Sample Beat Window")
        st.markdown(
            "The expected CSV contains one heartbeat segment: "
            "**90 samples before the R-peak** + **98 samples from the R-peak onward**."
        )

        # Render a synthetic example beat (purely illustrative)
        t    = np.linspace(-90, 98, 188)
        demo = (
            0.2 * np.exp(-0.5 * ((t + 30) / 8)  ** 2)   # P-wave
          + 1.5 * np.exp(-0.5 * ((t)       / 3)  ** 2)   # QRS peak (R)
          - 0.3 * np.exp(-0.5 * ((t - 4)   / 2)  ** 2)   # S-wave trough
          + 0.4 * np.exp(-0.5 * ((t - 40)  / 15) ** 2)   # T-wave
          + 0.02 * np.random.randn(188)                   # light noise
        )
        fig_demo, ax_demo = plt.subplots(figsize=(9, 2.8))
        fig_demo.patch.set_facecolor('#1a1a2e')
        ax_demo.set_facecolor('#16213e')
        ax_demo.plot(np.arange(188), demo, color='#40c4ff', linewidth=1.6)
        ax_demo.axvline(x=90, color='white', linestyle='--',
                        linewidth=1.0, alpha=0.5, label='R-peak (sample 90)')
        ax_demo.set_xlabel('Sample Index', color='#cccccc', fontsize=9)
        ax_demo.set_ylabel('Amplitude', color='#cccccc', fontsize=9)
        ax_demo.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax_demo.spines.values():
            spine.set_color('#333355')
        ax_demo.grid(True, linestyle=':', alpha=0.2, color='#aaaaaa')
        ax_demo.legend(loc='upper right', fontsize=8,
                       facecolor='#1a1a2e', edgecolor='#444466',
                       labelcolor='#cccccc')
        ax_demo.set_title(
            'Illustrative ECG Beat (not real data)',
            color='white', fontsize=10, pad=8
        )
        plt.tight_layout()
        st.pyplot(fig_demo, use_container_width=True)
        plt.close(fig_demo)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
