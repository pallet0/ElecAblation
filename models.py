"""SOGNN model (Li et al., 2021) for EEG emotion classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SOGC(nn.Module):
    """Self-Organized Graph Convolution layer.

    Bottleneck -> tanh -> softmax attention -> top-k sparsification ->
    self-loops -> graph convolution.
    """

    def __init__(self, n_electrodes, in_features, bn_features, out_features, top_k):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.top_k = min(top_k, n_electrodes)
        self.bn = nn.Linear(in_features, bn_features)
        self.gc = nn.Linear(in_features, out_features)

    def forward(self, x):
        B_E = x.shape[0]
        E = self.n_electrodes
        B = B_E // E

        h = x.reshape(B, E, -1)                                    # (B, E, C*H*W)

        g = torch.tanh(self.bn(h))                                # (B, E, bn)
        a = torch.softmax(g @ g.transpose(-1, -2), dim=-1)       # (B, E, E)

        vals, idxs = a.topk(self.top_k, dim=-1)                  # (B, E, k)
        a_sparse = torch.zeros_like(a).scatter_(-1, idxs, vals)   # (B, E, E)

        # Self-loops (diagonal = 1)
        eye = torch.eye(E, device=a.device, dtype=a.dtype).unsqueeze(0)
        a_sparse = a_sparse * (1 - eye) + eye

        out = torch.relu(self.gc(a_sparse @ h))                   # (B, E, out)
        return out


class SOGNN(nn.Module):
    """Self-Organized Graph Neural Network (Li et al., 2021).

    Interleaved CNN + SOGC with multi-scale parallel branches. Each CNN block
    feeds a SOGC branch; the three branch outputs are concatenated for
    classification.
    """

    def __init__(self, n_electrodes=62, n_bands=5, n_timeframes=64,
                 n_classes=4, top_k=10, dropout=0.1, **kwargs):
        super().__init__()
        self.n_electrodes = n_electrodes

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 5))

        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_bands, n_timeframes)
            d1 = self.pool(F.relu(self.conv1(dummy)))
            sogc1_in = d1.numel()       # 32 * 1 * 30 = 960 for (5, 64)
            d2 = self.pool(F.relu(self.conv2(d1)))
            sogc2_in = d2.numel()       # 64 * 1 * 13 = 832 for (5, 64)
            d3 = self.pool(F.relu(self.conv3(d2)))
            sogc3_in = d3.numel()       # 128 * 1 * 4 = 512 for (5, 64)

        self.sogc1 = SOGC(n_electrodes, sogc1_in, 64, 32, top_k)
        self.sogc2 = SOGC(n_electrodes, sogc2_in, 64, 32, top_k)
        self.sogc3 = SOGC(n_electrodes, sogc3_in, 64, 32, top_k)

        self.classifier = nn.Linear(n_electrodes * 32 * 3, n_classes)

    def forward(self, x):
        B, E = x.shape[0], x.shape[1]
        x = x.reshape(B * E, 1, x.shape[2], x.shape[3])  # per-electrode

        x = self.pool(self.drop(F.relu(self.conv1(x))))
        x1 = self.sogc1(x)                                # (B, E, 32)

        x = self.pool(self.drop(F.relu(self.conv2(x))))
        x2 = self.sogc2(x)                                # (B, E, 32)

        x = self.pool(self.drop(F.relu(self.conv3(x))))
        x3 = self.sogc3(x)                                # (B, E, 32)

        out = torch.cat([x1, x2, x3], dim=-1)             # (B, E, 96)
        out = self.drop(out)
        out = out.reshape(B, -1)                           # (B, E*96)
        return self.classifier(out)                        # (B, n_classes)
