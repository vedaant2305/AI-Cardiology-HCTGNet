# =============================================================================
# FILE: model.py
# PROJECT: Trustworthy Arrhythmia Diagnosis
# DESCRIPTION: Full PyTorch implementation of HCTG-Net — a Hybrid
#              CNN-Transformer Network with Gated Fusion for automatic
#              ECG arrhythmia classification (5 AAMI classes).
#
#              Architecture mirrors the paper exactly:
#                - Residual CNN Branch     → 256-dim vector  c
#                - Transformer Branch      → 128-dim vector  t
#                - Gated Fusion Module     → 256-dim vector  f
#                - Classifier Head         → 5-class logits
#
# REFERENCE: Xiong et al., "HCTG-Net: A Hybrid CNN–Transformer Network
#            with Gated Fusion for Automatic ECG Arrhythmia Diagnosis",
#            Bioengineering 2025, 12, 1268.
# =============================================================================

import math
import torch
import torch.nn as nn


# =============================================================================
# SECTION 1: CNN BRANCH COMPONENTS
# =============================================================================

class ResidualBlock1D(nn.Module):
    """
    A single 1-D Residual Block as used in the CNN branch of HCTG-Net.

    Structure (per block):
        Conv1d → BN → ReLU → Conv1d → BN → (+ skip) → ReLU

    The skip connection uses either:
        - An identity mapping  : when in_channels == out_channels AND stride == 1
        - A 1×1 Conv projection: when dimensions change (channel depth or stride),
          implemented as Wdown in Equation (2) of the paper.

    Args:
        in_channels  (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        stride       (int): Stride for the first conv layer. stride=2 halves
                            the temporal resolution (used in Residual Block 2).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # --- Main path: two stacked Conv1d layers ---
        self.conv_path = nn.Sequential(
            # First conv: may downsample temporally via stride
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            # Second conv: always stride=1, preserves resolution
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            # NOTE: final ReLU is applied AFTER the skip addition (see forward)
        )

        # --- Skip connection (Wdown in paper Equation 2) ---
        # A projection is needed whenever the skip tensor dimensions differ
        # from the conv_path output (different channels OR strided downsampling).
        if stride != 1 or in_channels != out_channels:
            self.skip_projection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            # Identity mapping — no learnable parameters needed
            self.skip_projection = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Equation (2) from the paper:
            y = ReLU( F(x) + Wdown(x) )

        Args:
            x: Input tensor of shape (B, in_channels, L)

        Returns:
            Tensor of shape (B, out_channels, L') where L' = L/stride
        """
        residual = self.skip_projection(x)   # skip branch: (B, C_out, L')
        out      = self.conv_path(x)         # main branch: (B, C_out, L')
        return self.relu(out + residual)     # element-wise add then activate


class CNNBranch(nn.Module):
    """
    Residual CNN Branch of HCTG-Net.

    Extracts LOCAL morphological features from the ECG waveform — QRS
    complex shape, P-wave and T-wave morphology, repolarisation patterns.

    Architecture (Table 1 of paper):
        Stem Conv  : Conv1d(1→64,  k=7, s=1, p=3) + BN + ReLU
        ResBlock 1 : (64  → 64),  stride=1  — preserves temporal resolution
        ResBlock 2 : (64  → 128), stride=2  — halves temporal resolution
        ResBlock 3 : (128 → 256), stride=1  — deepens channel representation
        AvgPool    : AdaptiveAvgPool1d(1)   — collapses time → scalar/channel
        Output     : 256-dim vector  c

    Args:
        in_channels (int): Input channels (1 for single-lead ECG).
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # --- Stem: large-kernel conv to capture broad initial features ---
        # kernel=7 gives a 7-sample receptive field at the first layer,
        # appropriate for detecting coarse waveform features before residual
        # refinement. padding=3 preserves the sequence length.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64,
                      kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # --- Residual Blocks ---
        # Block 1: 64→64,  stride=1 → output shape (B, 64,  L)
        self.res_block1 = ResidualBlock1D(in_channels=64,
                                          out_channels=64,
                                          stride=1)

        # Block 2: 64→128, stride=2 → output shape (B, 128, L/2)
        # Halving the temporal resolution doubles the effective receptive field
        # of subsequent layers, allowing coarser temporal pattern detection.
        self.res_block2 = ResidualBlock1D(in_channels=64,
                                          out_channels=128,
                                          stride=2)

        # Block 3: 128→256, stride=1 → output shape (B, 256, L/2)
        # Expands channel depth for richer feature representation.
        self.res_block3 = ResidualBlock1D(in_channels=128,
                                          out_channels=256,
                                          stride=1)

        # --- Global pooling: collapse the time dimension ---
        # AdaptiveAvgPool1d(1) reduces (B, 256, L') → (B, 256, 1)
        # regardless of the input temporal length. Followed by flatten → (B, 256)
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ECG input tensor of shape (B, 1, 188)

        Returns:
            c: Local morphology feature vector of shape (B, 256)
        """
        x = self.stem(x)        # (B, 1,   188) → (B, 64,  188)
        x = self.res_block1(x)  # (B, 64,  188) → (B, 64,  188)
        x = self.res_block2(x)  # (B, 64,  188) → (B, 128,  94)
        x = self.res_block3(x)  # (B, 128,  94) → (B, 256,  94)
        x = self.global_pool(x) # (B, 256,  94) → (B, 256,   1)
        c = x.squeeze(-1)       # (B, 256,   1) → (B, 256)
        return c


# =============================================================================
# SECTION 2: TRANSFORMER BRANCH COMPONENTS
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learnable) Sinusoidal Positional Encoding as defined in
    Vaswani et al. (2017) "Attention is All You Need" and used in HCTG-Net
    (Equation 3 of the paper):

        PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    This injects temporal position information into the embedded sequence,
    since self-attention is permutation-invariant by design and would
    otherwise be blind to the ordering of time-steps.

    Args:
        d_model  (int)  : Embedding dimension (must match linear embedding).
        max_len  (int)  : Maximum sequence length to pre-compute encodings for.
        dropout  (float): Dropout probability applied after adding PE.
    """

    def __init__(self, d_model: int = 128, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # --- Pre-compute the positional encoding matrix ---
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # position indices: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Division term: 10000^(2i/d_model), computed in log-space for
        # numerical stability.
        # Shape: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # Even indices  → sine
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices   → cosine
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension → (1, max_len, d_model) so it broadcasts
        # over batch during the forward pass.
        pe = pe.unsqueeze(0)

        # Register as a buffer: moves with the model (e.g. to GPU) but is
        # NOT a learnable parameter — it won't appear in optimizer.parameters().
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the embedded sequence.

        Args:
            x: Embedded sequence of shape (B, L, d_model)

        Returns:
            Tensor of shape (B, L, d_model) with positional info injected.
        """
        # self.pe[:, :x.size(1), :] slices the pre-computed table to the
        # actual sequence length L (which may be shorter than max_len).
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBranch(nn.Module):
    """
    Transformer Branch of HCTG-Net.

    Models GLOBAL temporal dependencies across the entire heartbeat sequence —
    rhythm irregularities, inter-beat timing context, and long-range waveform
    relationships that local convolutions cannot capture.

    Architecture (Table 1 of paper):
        Linear Embedding    : (B, 188, 1)   → (B, 188, 128)
        Positional Encoding : sinusoidal, d=128
        Transformer × 2     : MHSA (4 heads) + FFN (dim=256) + residuals
        AdaptiveAvgPool1d   : collapse sequence → (B, 128)
        Output              : 128-dim vector  t

    Args:
        seq_len   (int): Input sequence length (188 samples per beat).
        d_model   (int): Embedding / model dimension.
        n_heads   (int): Number of attention heads in MHSA.
        ffn_dim   (int): Inner dimension of the position-wise FFN.
        n_layers  (int): Number of stacked Transformer encoder blocks.
        dropout   (float): Dropout probability used throughout.
    """

    def __init__(self,
                 seq_len : int   = 188,
                 d_model : int   = 128,
                 n_heads : int   = 4,
                 ffn_dim : int   = 256,
                 n_layers: int   = 2,
                 dropout : float = 0.1):
        super().__init__()

        # --- Step 1: Linear patch embedding ---
        # Projects each scalar time-step value (1-D) to a d_model-dimensional
        # token embedding. Input arrives as (B, 1, 188); we first permute to
        # (B, 188, 1) to treat each time-step as a token, then apply the linear.
        self.embedding = nn.Linear(in_features=1, out_features=d_model)

        # --- Step 2: Sinusoidal positional encoding ---
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=seq_len + 16,  # +16 margin in case seq_len varies slightly
            dropout=dropout,
        )

        # --- Step 3: Stack of Transformer Encoder blocks ---
        # PyTorch's TransformerEncoderLayer packages MHSA + FFN + residuals
        # + layer norm in a single module, matching the paper's architecture.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,   # inner FFN width = 256
            dropout=dropout,
            activation='relu',
            batch_first=True,          # CRITICAL: expects (B, L, d_model),
                                       # not PyTorch's default (L, B, d_model)
            norm_first=False,          # Post-LN: matches paper's "Add & Norm"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,       # 2 stacked blocks
        )

        # --- Step 4: Global average pool over sequence length ---
        # Collapses (B, 188, 128) → (B, 128, 1) → (B, 128) via squeeze.
        # We permute before pooling so the time axis is the last dimension,
        # as required by AdaptiveAvgPool1d.
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ECG input tensor of shape (B, 1, 188)

        Returns:
            t: Global temporal feature vector of shape (B, 128)
        """
        # (B, 1, 188) → (B, 188, 1): place time-steps on dim=1 for embedding
        x = x.permute(0, 2, 1)

        # Linear embedding: (B, 188, 1) → (B, 188, 128)
        x = self.embedding(x)

        # Add sinusoidal positional encoding: (B, 188, 128) → (B, 188, 128)
        x = self.pos_encoding(x)

        # Self-attention + FFN × 2: (B, 188, 128) → (B, 188, 128)
        x = self.transformer_encoder(x)

        # Permute for pooling: (B, 188, 128) → (B, 128, 188)
        x = x.permute(0, 2, 1)

        # Global pool: (B, 128, 188) → (B, 128, 1) → (B, 128)
        x = self.global_pool(x)
        t = x.squeeze(-1)
        return t


# =============================================================================
# SECTION 3: GATED FUSION MODULE
# =============================================================================

class GatedFusionModule(nn.Module):
    """
    Gated Fusion Module — the core novelty of HCTG-Net.

    Adaptively balances CNN-derived LOCAL morphology features (c) and
    Transformer-derived GLOBAL temporal features (t) on a per-dimension
    basis. Unlike simple concatenation or addition, the gate g learns to
    weight each of the 256 latent dimensions independently, allowing the
    model to emphasise morphology-dominant cues for ventricular beats and
    context-dominant cues for supraventricular patterns.

    Equations (5), (6), (7) from the paper:

        c_tilde = Wc @ c                     — project CNN features to H=256
        t_tilde = Wt @ t                     — project Transformer features to H=256
        g       = σ( MLP([c_tilde; t_tilde]) )  — gate: (512→256), per-dim sigmoid
        f       = g ⊙ c_tilde + (1-g) ⊙ t_tilde  — soft weighted fusion

    Args:
        cnn_dim    (int): Dimensionality of CNN feature vector c (256).
        trans_dim  (int): Dimensionality of Transformer feature vector t (128).
        hidden_dim (int): Shared projection & fusion dimension H (256).
    """

    def __init__(self,
                 cnn_dim   : int = 256,
                 trans_dim : int = 128,
                 hidden_dim: int = 256):
        super().__init__()

        # --- Equation (5): Project both branches to the shared space H ---
        # Wc: maps CNN output c (256-dim) → c_tilde (256-dim)
        self.proj_cnn = nn.Linear(cnn_dim, hidden_dim)

        # Wt: maps Transformer output t (128-dim) → t_tilde (256-dim)
        self.proj_trans = nn.Linear(trans_dim, hidden_dim)

        # --- Equation (6): Gating network g = σ( MLP([c_tilde; t_tilde]) ) ---
        # Input to MLP is the concatenation of both projections → 2H = 512-dim.
        # The MLP projects 512 → 256 through two FC layers (as per Table 1),
        # and sigmoid squashes each of the 256 outputs to (0, 1).
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 512 → 256
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),       # 256 → 256
            nn.Sigmoid(),                            # per-dimension gate ∈ (0,1)
        )

    def forward(self, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: CNN feature vector,         shape (B, 256)
            t: Transformer feature vector, shape (B, 128)

        Returns:
            f: Fused feature vector,       shape (B, 256)
        """
        # --- Equation (5): project to shared space ---
        c_tilde = self.proj_cnn(c)     # (B, 256) → (B, 256)
        t_tilde = self.proj_trans(t)   # (B, 128) → (B, 256)

        # --- Equation (6): compute per-dimension gate ---
        # Concatenate along the feature dimension: (B, 256) + (B, 256) → (B, 512)
        combined = torch.cat([c_tilde, t_tilde], dim=-1)  # (B, 512)
        g = self.gate_mlp(combined)                        # (B, 256), values ∈ (0,1)

        # --- Equation (7): soft gated fusion ---
        # For each of the 256 dimensions independently:
        #   - If g[i] ≈ 1.0 → emphasise CNN  (morphology-dominant)
        #   - If g[i] ≈ 0.0 → emphasise Transformer (context-dominant)
        f = g * c_tilde + (1.0 - g) * t_tilde  # (B, 256)

        return f


# =============================================================================
# SECTION 4: CLASSIFIER HEAD
# =============================================================================

class ClassifierHead(nn.Module):
    """
    Final classification head applied to the fused feature vector f.

    Architecture (Table 1 of paper):
        FC(256 → 128) → ReLU → Dropout(0.3) → FC(128 → 5)

    Note: The output is raw LOGITS (no softmax applied here).
    Softmax is omitted because:
        - nn.CrossEntropyLoss applies log-softmax + NLL internally,
          which is numerically more stable than computing softmax then log.
        - During inference, argmax on logits == argmax on probabilities.
        - If calibrated probabilities are needed, apply torch.softmax()
          externally at inference time.

    Args:
        in_features  (int): Dimensionality of the fused feature vector (256).
        num_classes  (int): Number of output classes (5 for AAMI standard).
        dropout_rate (float): Dropout probability for regularisation.
    """

    def __init__(self,
                 in_features : int   = 256,
                 num_classes : int   = 5,
                 dropout_rate: float = 0.3):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features, 128),   # FC: 256 → 128
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),    # regularisation
            nn.Linear(128, num_classes),   # FC: 128 → 5  (logits)
            # NO softmax here — CrossEntropyLoss handles it internally
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: Fused feature vector of shape (B, 256)

        Returns:
            logits: Raw class scores of shape (B, 5)
        """
        return self.head(f)


# =============================================================================
# SECTION 5: FULL HCTG-NET MODEL
# =============================================================================

class HCTGNet(nn.Module):
    """
    HCTG-Net: Hybrid CNN–Transformer Network with Gated Fusion.

    Combines three complementary modules into a unified architecture for
    automatic ECG arrhythmia classification:

        1. CNN Branch       — captures local waveform morphology
        2. Transformer Branch — captures global temporal dependencies
        3. Gated Fusion     — adaptively integrates both representations

    Input shape  : (B, 1, 188)  — batch of single-channel 188-sample heartbeats
    Output shape : (B, 5)       — raw logits for 5 AAMI arrhythmia classes

    AAMI class mapping:
        0 → N (Normal / Bundle Branch Block / Escape)
        1 → S (Supraventricular ectopic)
        2 → V (Ventricular ectopic)
        3 → F (Fusion)
        4 → Q (Unknown / Paced)

    Args:
        num_classes  (int)  : Number of output classes (default: 5).
        d_model      (int)  : Transformer embedding dimension (default: 128).
        n_heads      (int)  : MHSA attention heads (default: 4).
        ffn_dim      (int)  : Transformer FFN inner dimension (default: 256).
        n_layers     (int)  : Number of Transformer blocks (default: 2).
        dropout      (float): Dropout for Transformer and Classifier (default: 0.1).
        clf_dropout  (float): Dropout inside the Classifier head (default: 0.3).
    """

    def __init__(self,
                 num_classes : int   = 5,
                 d_model     : int   = 128,
                 n_heads     : int   = 4,
                 ffn_dim     : int   = 256,
                 n_layers    : int   = 2,
                 dropout     : float = 0.1,
                 clf_dropout : float = 0.3):
        super().__init__()

        # --- Branch 1: Residual CNN ---
        # Outputs c ∈ R^256 encoding local morphological features.
        self.cnn_branch = CNNBranch(in_channels=1)

        # --- Branch 2: Transformer Encoder ---
        # Outputs t ∈ R^128 encoding global temporal context.
        self.transformer_branch = TransformerBranch(
            seq_len=188,
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        # --- Gated Fusion Module ---
        # Combines c and t into fused vector f ∈ R^256 via learnable gating.
        self.fusion = GatedFusionModule(
            cnn_dim=256,       # CNN output dim
            trans_dim=d_model, # Transformer output dim (128)
            hidden_dim=256,    # shared projection space H
        )

        # --- Classifier Head ---
        # Maps f → 5 class logits.
        self.classifier = ClassifierHead(
            in_features=256,
            num_classes=num_classes,
            dropout_rate=clf_dropout,
        )

        # --- Weight initialisation ---
        # Apply He (Kaiming) initialisation to Conv1d and Linear layers.
        # This is standard practice for ReLU networks and improves convergence.
        self._initialise_weights()

    def _initialise_weights(self):
        """
        Applies He (Kaiming) uniform initialisation to Conv1d and Linear
        layers, and sets BatchNorm weights to 1 and biases to 0.
        This mirrors common ResNet initialisation practice.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight,
                                         mode='fan_out',
                                         nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight,
                                         mode='fan_in',
                                         nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass of HCTG-Net.

        Args:
            x: Input ECG tensor of shape (B, 1, 188)

        Returns:
            logits: Raw class scores of shape (B, 5).
                    Pass through torch.softmax(logits, dim=-1) at inference
                    time to obtain class probabilities.
        """
        # --- Branch 1: local morphological features ---
        # (B, 1, 188) → (B, 256)
        c = self.cnn_branch(x)

        # --- Branch 2: global temporal context ---
        # (B, 1, 188) → (B, 128)
        t = self.transformer_branch(x)

        # --- Gated fusion: adaptive per-dimension combination ---
        # (B, 256) + (B, 128) → (B, 256)
        f = self.fusion(c, t)

        # --- Classification: fused features → class logits ---
        # (B, 256) → (B, 5)
        logits = self.classifier(f)

        return logits

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# SECTION 6: SMOKE TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  HCTG-Net Architecture — Smoke Test")
    print("=" * 60 + "\n")

    # Reproducibility
    torch.manual_seed(42)

    # --- Instantiate the model ---
    model = HCTGNet(
        num_classes=5,
        d_model=128,
        n_heads=4,
        ffn_dim=256,
        n_layers=2,
        dropout=0.1,
        clf_dropout=0.3,
    )
    model.eval()  # disable dropout for the shape test

    # --- Create a dummy batch: 16 heartbeats, 1 channel, 188 samples ---
    dummy_input = torch.randn(16, 1, 188)
    print(f"  Input  shape : {list(dummy_input.shape)}")

    # --- Forward pass (no gradients needed for shape check) ---
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"  Output shape : {list(logits.shape)}")
    print(f"  Output dtype : {logits.dtype}")

    # --- Shape assertion ---
    assert logits.shape == (16, 5), \
        f"Expected output shape (16, 5), got {tuple(logits.shape)}"
    print("\n  ✓ Output shape (16, 5) confirmed.\n")

    # --- Print per-module intermediate shapes for architecture verification ---
    print("  Intermediate tensor shapes:")
    with torch.no_grad():
        c = model.cnn_branch(dummy_input)
        t = model.transformer_branch(dummy_input)
        f = model.fusion(c, t)

    print(f"    CNN Branch output        c  : {list(c.shape)}")
    print(f"    Transformer Branch output t  : {list(t.shape)}")
    print(f"    Gated Fusion output       f  : {list(f.shape)}")
    print(f"    Classifier output  (logits)  : {list(logits.shape)}")

    # --- Parameter count ---
    n_params = model.count_parameters()
    print(f"\n  Total trainable parameters : {n_params:,}")

    print("\n" + "=" * 60)
    print("  ✓ All checks passed. Model is ready for training.")
    print("=" * 60 + "\n")
