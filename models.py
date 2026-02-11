"""Model definitions for EEG electrode ablation study."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SOGC(nn.Module):
    """Self-Organized Graph Convolution layer (Li et al., 2021).

    Computes a sparse adjacency matrix from node features via bottleneck
    projection + tanh + softmax + top-k, adds self-loops (matching
    DenseGCNConv's add_loop=True default), then applies graph convolution.

    Args:
        n_electrodes: number of graph nodes (electrodes)
        in_features: flattened feature dim per node (C*H*W from CNN)
        bn_features: bottleneck dim for adjacency computation
        out_features: output feature dim per node
        top_k: number of neighbors to keep in sparse adjacency
    """

    def __init__(self, n_electrodes, in_features, bn_features, out_features, top_k):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.top_k = min(top_k, n_electrodes)
        self.bn = nn.Linear(in_features, bn_features)
        self.gc = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x: (B*E, C, H, W) feature maps from CNN stage
        Returns:
            (B, E, out_features) graph-convolved node features
        """
        B_E = x.shape[0]
        E = self.n_electrodes
        B = B_E // E

        # Flatten spatial dims: (B*E, C, H, W) -> (B, E, C*H*W)
        h = x.reshape(B, E, -1)

        # Compute adjacency: bottleneck -> tanh -> softmax -> top-k
        g = torch.tanh(self.bn(h))                                # (B, E, bn)
        a = torch.softmax(g @ g.transpose(-1, -2), dim=-1)       # (B, E, E)

        # Top-k sparsification
        vals, idxs = a.topk(self.top_k, dim=-1)                  # (B, E, k)
        a_sparse = torch.zeros_like(a).scatter_(-1, idxs, vals)   # (B, E, E)

        # Add self-loops: set diagonal to 1.0
        eye = torch.eye(E, device=a.device, dtype=a.dtype).unsqueeze(0)
        a_sparse = a_sparse * (1 - eye) + eye

        # Graph convolution
        out = torch.relu(self.gc(a_sparse @ h))                   # (B, E, out)
        return out


class SOGNN(nn.Module):
    """Self-Organized Graph Neural Network for EEG emotion classification.

    Architecture (Li et al., 2021) — interleaved CNN + SOGC with multi-scale
    parallel branches:

      Input (B, E, 5, 64) -> reshape (B*E, 1, 5, 64)
        -> Conv1 -> ReLU -> Drop -> Pool -> SOGC1 -> x1 (B, E, 32)
        -> Conv2 -> ReLU -> Drop -> Pool -> SOGC2 -> x2 (B, E, 32)
        -> Conv3 -> ReLU -> Drop -> Pool -> SOGC3 -> x3 (B, E, 32)
        -> cat([x1, x2, x3]) -> Drop -> flatten -> Linear -> logits

    Args:
        n_electrodes: number of EEG electrodes (62 default, variable for retrain ablation)
        n_bands: number of frequency bands (5 for DE features)
        n_timeframes: temporal padding length (64 default)
        n_classes: number of output classes (4 for SEED-IV)
        top_k: number of neighbors to keep in sparse adjacency
        dropout: dropout rate
        **kwargs: absorbs extra keys
    """

    def __init__(self, n_electrodes=62, n_bands=5, n_timeframes=64,
                 n_classes=4, top_k=10, dropout=0.1, **kwargs):
        super().__init__()
        self.n_electrodes = n_electrodes

        # CNN blocks (applied per-electrode)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 5))

        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop = nn.Dropout(dropout)

        # Compute intermediate feature dimensions dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_bands, n_timeframes)
            d1 = self.pool(F.relu(self.conv1(dummy)))
            sogc1_in = d1.numel()       # 32 * 1 * 30 = 960 for (5, 64)
            d2 = self.pool(F.relu(self.conv2(d1)))
            sogc2_in = d2.numel()       # 64 * 1 * 13 = 832 for (5, 64)
            d3 = self.pool(F.relu(self.conv3(d2)))
            sogc3_in = d3.numel()       # 128 * 1 * 4 = 512 for (5, 64)

        # SOGC branches (multi-scale graph reasoning)
        self.sogc1 = SOGC(n_electrodes, sogc1_in, 64, 32, top_k)
        self.sogc2 = SOGC(n_electrodes, sogc2_in, 64, 32, top_k)
        self.sogc3 = SOGC(n_electrodes, sogc3_in, 64, 32, top_k)

        # Classifier: multi-scale concat -> logits
        self.classifier = nn.Linear(n_electrodes * 32 * 3, n_classes)

    def forward(self, x):
        """
        Args:
            x: (B, n_electrodes, n_bands, n_timeframes)
        Returns:
            logits: (B, n_classes)
        """
        B, E = x.shape[0], x.shape[1]

        # Reshape to per-electrode: (B*E, 1, bands, T)
        x = x.reshape(B * E, 1, x.shape[2], x.shape[3])

        # Block 1 + SOGC1 branch
        x = self.pool(self.drop(F.relu(self.conv1(x))))
        x1 = self.sogc1(x)                              # (B, E, 32)

        # Block 2 + SOGC2 branch
        x = self.pool(self.drop(F.relu(self.conv2(x))))
        x2 = self.sogc2(x)                              # (B, E, 32)

        # Block 3 + SOGC3 branch
        x = self.pool(self.drop(F.relu(self.conv3(x))))
        x3 = self.sogc3(x)                              # (B, E, 32)

        # Multi-scale concat + classify
        out = torch.cat([x1, x2, x3], dim=-1)           # (B, E, 96)
        out = self.drop(out)
        out = out.reshape(B, -1)                         # (B, E * 96)
        logits = self.classifier(out)                    # (B, n_classes)
        return logits
