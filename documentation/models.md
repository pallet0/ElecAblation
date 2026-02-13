# Model Architecture (`models.py`)

This project utilizes the **Self-Organized Graph Neural Network (SOGNN)**, as proposed by Li et al. (2021), adapted for electrode ablation studies.

## SOGC: Self-Organized Graph Convolution

The core of SOGNN is the `SOGC` layer, which learns a dynamic graph structure for every input sample rather than using a fixed, distance-based adjacency matrix.

```python
class SOGC(nn.Module):
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

        # [Line 1] Flatten spatial dims: (B*E, C, H, W) -> (B, E, C*H*W)
        h = x.reshape(B, E, -1)

        # [Line 2] Compute adjacency: bottleneck -> tanh -> softmax -> top-k
        g = torch.tanh(self.bn(h))                                # (B, E, bn)
        a = torch.softmax(g @ g.transpose(-1, -2), dim=-1)       # (B, E, E)

        # [Line 3] Top-k sparsification
        vals, idxs = a.topk(self.top_k, dim=-1)                  # (B, E, k)
        a_sparse = torch.zeros_like(a).scatter_(-1, idxs, vals)   # (B, E, E)

        # [Line 4] Add self-loops: set diagonal to 1.0
        eye = torch.eye(E, device=a.device, dtype=a.dtype).unsqueeze(0)
        a_sparse = a_sparse * (1 - eye) + eye

        # [Line 5] Graph convolution
        out = torch.relu(self.gc(a_sparse @ h))                   # (B, E, out)
        return out
```

### Line-by-Line Breakdown (SOGC)
- **Line 1 (`h = x.reshape(...)`)**: Flattens the spatial/spectral features (Channels × Height × Width) into a single vector per electrode. This prepares the data for graph-level operations where each electrode is a node.
- **Line 2 (`g = ...`, `a = ...`)**: `bn` projects features into a lower-dimensional bottleneck space to find relationships more efficiently. `g @ g.transpose` calculates the similarity between every electrode pair, and `softmax` ensures the total weight from one electrode to others sums to 1.
- **Line 3 (`a_sparse = ...`)**: Only the `top_k` strongest connections are kept. `scatter_` places the original softmax values back into a zeroed matrix at the top-k indices, creating a sparse adjacency.
- **Line 4 (`a_sparse * (1 - eye) + eye`)**: Sets the diagonal of the adjacency matrix to exactly 1.0. This ensures that during message passing, an electrode's own features are always preserved.
- **Line 5 (`torch.relu(self.gc(a_sparse @ h))`)**: Performs the graph convolution. `a_sparse @ h` aggregates features from neighboring electrodes, and `self.gc` (a linear layer) applies the learnable transformation.

## SOGNN: System Integration

The `SOGNN` model combines a 2D CNN backbone with multiple `SOGC` branches to capture multi-scale spatial and temporal features.

```python
    def forward(self, x):
        B, E = x.shape[0], x.shape[1]

        # [Line 6] Reshape to per-electrode: (B*E, 1, bands, T)
        x = x.reshape(B * E, 1, x.shape[2], x.shape[3])

        # [Line 7] Block 1 + SOGC1 branch
        x = self.pool(self.drop(F.relu(self.conv1(x))))
        x1 = self.sogc1(x)                              # (B, E, 32)

        # [Line 8] Multi-scale concat + classify
        out = torch.cat([x1, x2, x3], dim=-1)           # (B, E, 96)
        out = self.drop(out)
        out = out.reshape(B, -1)                         # (B, E * 96)
        logits = self.classifier(out)                    # (B, n_classes)
        return logits
```

### Line-by-Line Breakdown (SOGNN)
- **Line 6 (`x.reshape(...)`)**: Collapses the batch and electrode dimensions. This allows the 2D CNN to treat every single electrode from every trial as an independent image, ensuring the CNN only learns spectral-temporal patterns *within* an electrode, not across them.
- **Line 7 (`x1 = self.sogc1(x)`)**: After CNN processing, the features are passed to the SOGC layer. SOGC internally restores the `(B, E)` structure to learn relationships *between* the processed electrode features.
- **Line 8 (`torch.cat(...)`, `out.reshape(...)`)**: Concatenates features from low, mid, and high levels of the CNN. The final `reshape(B, -1)` flattens all electrodes and their multi-scale features into one long vector for the final classification decision.
