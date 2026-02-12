# Model Architecture (`models.py`)

This project utilizes the **Self-Organized Graph Neural Network (SOGNN)**, as proposed by Li et al. (2021), adapted for electrode ablation studies.

## SOGC: Self-Organized Graph Convolution

The core of SOGNN is the SOGC layer, which learns a dynamic graph structure for every input sample rather than using a fixed, distance-based adjacency matrix.

### Adjacency Learning Mechanism
1. **Bottleneck Projection**: Node features are projected into a lower-dimensional space using a linear layer and a `tanh` activation.
2. **Similarity Computation**: A similarity matrix is computed via a batched dot-product: $A = \text{softmax}(G \cdot G^T)$.
3. **Top-k Sparsification**: For each electrode, only the $k$ most significant connections (default $k=10$) are kept. This ensures the graph remains sparse and focuses on the most relevant electrode-to-electrode relationships.
4. **Self-Loops**: Diagonal elements are set to 1.0 to ensure nodes retain their own feature information during message passing.

### Graph Convolution
The convolution follows the standard formulation: $H' = \text{ReLU}(A \cdot H \cdot W)$, where $A$ is the learned sparse adjacency and $W$ is the weight matrix.

## SOGNN: System Integration

The SOGNN model combines a 2D CNN backbone with multiple SOGC branches to capture multi-scale spatial and temporal features.

### 1. Per-Electrode CNN Processing
Before any graph operations, each electrode's $(Bands \times Time)$ map is processed independently by a series of convolutional and max-pooling layers. This prevents feature mixing between electrodes in the early stages and allows the model to extract clean spectral-temporal features.

### 2. Multi-Scale Parallel Branches
SOGNN uses three parallel branches that tap into different depths of the CNN backbone:
- **Branch 1**: Low-level features (from `conv1`).
- **Branch 2**: Mid-level features (from `conv2`).
- **Branch 3**: High-level features (from `conv3`).

Each branch passes its features through a dedicated SOGC layer, allowing the model to reason about electrode relationships at different levels of abstraction.

### 3. Feature Fusion
The outputs from all three SOGC branches are concatenated and flattened. This result is passed through a final linear classifier to produce the 4 emotion logits.

## Design for Ablation

The model is designed to be **electrode-count agnostic**:
- **Dynamic Dimensioning**: During initialization, a dummy forward pass is used to calculate the flatten dimensions, ensuring the model can be instantiated with any number of electrodes (e.g., for retrain-from-scratch ablation).
- **Masking Compatibility**: The forward pass accepts 4D tensors, allowing the pipeline to apply binary masks to specific channels without altering the underlying graph construction logic for the remaining channels.
