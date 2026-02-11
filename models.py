"""Model definitions for EEG electrode ablation study."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelBN(nn.Module):
    """BatchNorm1d for (B, C, D) tensors — normalizes each D feature across B*C."""
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)
    def forward(self, x):
        B, C, D = x.shape
        return self.bn(x.reshape(B * C, D)).reshape(B, C, D)


class ChannelAttentionEEGNet(nn.Module):
    """Shared spectral encoder + Bahdanau attention + classifier.

    Input:  (B, C, n_bands)
    Output: (logits (B, n_classes), alpha (B, C))
    """

    def __init__(self, n_bands=5, n_classes=4, d_hidden=64, dropout=0.5,
                 n_channels=62):
        super().__init__()
        self.spectral_encoder = nn.Sequential(
            nn.Linear(n_bands, d_hidden),
            _ChannelBN(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            _ChannelBN(d_hidden),
            nn.GELU(),
        )
        self.channel_embedding = nn.Parameter(
            torch.randn(1, n_channels, d_hidden) * 0.02)
        self.attn_scorer = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.Tanh(),
            nn.Linear(d_hidden // 2, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x, channel_mask=None):
        h = self.spectral_encoder(x) + self.channel_embedding  # (B, C, d_hidden)
        # Bahdanau attention pooling
        e = self.attn_scorer(h).squeeze(-1)         # (B, C)
        if channel_mask is not None:
            e = e.masked_fill(channel_mask == 0, float('-inf'))
        alpha = F.softmax(e, dim=-1)                # (B, C)
        context = torch.einsum('bc,bcd->bd', alpha, h)
        logits = self.classifier(context)
        return logits, alpha

    @torch.no_grad()
    def get_channel_importance(self, dataloader, device='cuda'):
        self.eval()
        all_alpha = []
        for x, _ in dataloader:
            x = x.to(device)
            _, alpha = self.forward(x)
            all_alpha.append(alpha.cpu())
        mean_alpha = torch.cat(all_alpha, dim=0).mean(dim=0)
        ranking = mean_alpha.argsort(descending=True)
        return mean_alpha.numpy(), ranking.numpy()


class MLPBaseline(nn.Module):
    """310 -> 4 MLP with BatchNorm + ReLU + Dropout."""

    def __init__(self, input_dim=310, n_classes=4, h1=128, h2=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class DualAttentionEEGNet(nn.Module):
    """Band attention -> Channel attention -> Classifier.

    Input:  (B, C, n_bands)
    Output: (logits (B, n_classes), channel_alpha (B, C), band_alpha (B, C, n_bands))
    """

    def __init__(self, n_bands=5, n_classes=4, d_band=16, d_chan=64, dropout=0.5):
        super().__init__()
        self.band_embed = nn.Sequential(
            nn.Linear(1, d_band),
            nn.GELU(),
        )
        self.band_attn = nn.Sequential(
            nn.Linear(d_band, d_band // 2),
            nn.Tanh(),
            nn.Linear(d_band // 2, 1, bias=False),
        )
        self.channel_proj = nn.Sequential(
            nn.Linear(d_band, d_chan),
            nn.LayerNorm(d_chan),
            nn.GELU(),
        )
        self.chan_attn = nn.Sequential(
            nn.Linear(d_chan, d_chan // 2),
            nn.Tanh(),
            nn.Linear(d_chan // 2, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_chan),
            nn.Linear(d_chan, d_chan),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_chan, n_classes),
        )

    def forward(self, x, channel_mask=None):
        B, C, nb = x.shape
        h_band = self.band_embed(x.unsqueeze(-1))      # (B, C, nb, d_band)
        e_band = self.band_attn(h_band).squeeze(-1)     # (B, C, nb)
        beta = F.softmax(e_band, dim=-1)                 # (B, C, nb)
        h_chan = torch.einsum('bcn,bcnd->bcd', beta, h_band)  # (B, C, d_band)
        h_chan = self.channel_proj(h_chan)                # (B, C, d_chan)
        e_chan = self.chan_attn(h_chan).squeeze(-1)        # (B, C)
        if channel_mask is not None:
            e_chan = e_chan.masked_fill(channel_mask == 0, float('-inf'))
        alpha = F.softmax(e_chan, dim=-1)                # (B, C)
        context = torch.einsum('bc,bcd->bd', alpha, h_chan)
        logits = self.classifier(context)
        return logits, alpha, beta


class SOGNN(nn.Module):
    """Self-Organized Graph Neural Network for EEG emotion classification.

    Architecture (Li et al., 2021):
      1. Per-electrode conv-pool chain: (B, E, 5, T) -> (B, E, 512) node features
      2. Three self-organized graph convolution (SOGC) layers: 512->64->64->64
      3. Flatten + FC: (B, E*64) -> (B, n_classes)

    Args:
        n_electrodes: number of EEG electrodes (62 default, variable for retrain ablation)
        n_bands: number of frequency bands (5 for DE features)
        n_timeframes: temporal padding length (64 default)
        n_classes: number of output classes (4 for SEED-IV)
        top_k: number of neighbors to keep in sparse adjacency
        dropout: dropout rate for conv blocks and FC layer
        **kwargs: absorbs extra keys from HP search dicts
    """

    def __init__(self, n_electrodes=62, n_bands=5, n_timeframes=64,
                 n_classes=4, top_k=10, dropout=0.1, **kwargs):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.top_k = min(top_k, n_electrodes)

        # ── Per-electrode conv-pool chain ──
        # Input per electrode: (1, 5, T) -> after 3 blocks -> (128, 1, T')
        self.conv_pool = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(64, 128, kernel_size=(1, 5), padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # Compute conv-pool output dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_bands, n_timeframes)
            conv_out = self.conv_pool(dummy)
            self._node_feat_dim = conv_out.numel()  # 128 * 1 * 4 = 512 for (5, 64)

        # ── SOGC layers (3 sequential) ──
        d_bn = 32  # bottleneck dim for adjacency computation
        # Layer 1: 512 -> 64
        self.W_bn1 = nn.Linear(self._node_feat_dim, d_bn, bias=False)
        self.gc1 = nn.Linear(self._node_feat_dim, 64)
        # Layer 2: 64 -> 64
        self.W_bn2 = nn.Linear(64, d_bn, bias=False)
        self.gc2 = nn.Linear(64, 64)
        # Layer 3: 64 -> 64
        self.W_bn3 = nn.Linear(64, d_bn, bias=False)
        self.gc3 = nn.Linear(64, 64)

        # ── Flatten + FC ──
        self._graph_out_dim = 64
        self.classifier = nn.Linear(n_electrodes * self._graph_out_dim, n_classes)

    def _sogc(self, H, W_bn, gc_layer):
        """One self-organized graph convolution step.

        Args:
            H: node features (B, E, D_in)
            W_bn: bottleneck linear layer (D_in -> d_bn)
            gc_layer: graph conv linear layer (D_in -> D_out)
        Returns:
            H_out: (B, E, D_out)
        """
        G = torch.tanh(W_bn(H))                                # (B, E, d_bn)
        A = torch.softmax(G @ G.transpose(-1, -2), dim=-1)     # (B, E, E)
        # Top-k sparsification (no re-normalization)
        vals, idxs = A.topk(self.top_k, dim=-1)                # (B, E, k)
        A_sparse = torch.zeros_like(A).scatter_(-1, idxs, vals) # (B, E, E)
        H_out = torch.relu(gc_layer(A_sparse @ H))             # (B, E, D_out)
        return H_out

    def forward(self, x):
        """
        Args:
            x: (B, n_electrodes, n_bands, n_timeframes)
        Returns:
            logits: (B, n_classes)
        """
        B, E = x.shape[0], x.shape[1]

        # Per-electrode conv-pool: reshape to (B*E, 1, bands, T)
        x = x.reshape(B * E, 1, x.shape[2], x.shape[3])
        x = self.conv_pool(x)                        # (B*E, 128, 1, T')
        x = x.reshape(B, E, -1)                      # (B, E, 512) node features

        # 3 SOGC layers
        x = self._sogc(x, self.W_bn1, self.gc1)      # (B, E, 64)
        x = self._sogc(x, self.W_bn2, self.gc2)      # (B, E, 64)
        x = self._sogc(x, self.W_bn3, self.gc3)      # (B, E, 64)

        # Flatten + classify
        x = x.reshape(B, -1)                          # (B, E * 64)
        logits = self.classifier(x)                   # (B, n_classes)
        return logits
