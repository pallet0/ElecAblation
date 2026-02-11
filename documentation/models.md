# models.md Documentation

This file defines the Deep Neural Network architectures used for EEG emotion classification, specifically the Self-Organized Graph Neural Network (SOGNN).

## `SOGC` (Self-Organized Graph Convolution) Class

The `SOGC` layer dynamically learns a sparse adjacency matrix from electrode features instead of using a fixed graph structure.

### Initialization (`__init__`)
- `n_electrodes`: Number of nodes (electrodes) in the graph.
- `in_features`: Dimension of the input features for each node.
- `bn_features`: Dimension of the bottleneck layer used to compute the adjacency matrix.
- `out_features`: Dimension of the output features after graph convolution.
- `top_k`: The number of neighbors to retain for each node (enforces sparsity).
- `self.bn`: A linear layer that projects node features into a bottleneck space for similarity computation.
- `self.gc`: A linear layer that performs the graph convolution (message passing).

### Forward Pass (`forward`)
1. **Feature Reshaping**: Reshapes input `x` (B*E, C, H, W) into (B, E, Features).
2. **Adjacency Computation**:
   - Projects features through a bottleneck and applies `tanh`.
   - Computes a self-attention-like matrix using dot-product similarity (`g @ g^T`).
   - Applies `softmax` to normalize the connections.
3. **Sparsification**: Keeps only the `top_k` strongest connections for each node using `topk` and `scatter_`.
4. **Self-loops**: Adds self-loops by setting the diagonal of the adjacency matrix to 1.0.
5. **Graph Convolution**: Performs the operation $Y = 	ext{ReLU}(A \cdot X \cdot W)$, where $A$ is the learned adjacency matrix.

---

## `SOGNN` (Self-Organized Graph Neural Network) Class

The main model architecture that combines multi-scale CNN branches with SOGC layers.

### Initialization (`__init__`)
- **CNN Blocks**: Three convolutional layers (`conv1`, `conv2`, `conv3`) with different kernel sizes to extract features at multiple temporal/spectral scales.
- **Dynamic Dimension Calculation**: Uses a dummy forward pass to determine the input sizes for subsequent SOGC layers.
- **SOGC Branches**: Three parallel SOGC layers (`sogc1`, `sogc2`, `sogc3`) that process features from different depths of the CNN backbone.
- **Classifier**: A final linear layer that concatenates the outputs from all three SOGC branches to produce class logits.

### Forward Pass (`forward`)
1. **Input Reshaping**: Reshapes the input EEG data (Batch, Electrodes, Bands, Time) into a per-electrode format for the CNN stage.
2. **Multi-scale Feature Extraction**:
   - **Branch 1**: Passes data through `conv1`, `ReLU`, `dropout`, and `pool`, then feeds the result into `sogc1`.
   - **Branch 2**: Continues from the previous CNN output through `conv2`, `ReLU`, `dropout`, and `pool`, then feeds the result into `sogc2`.
   - **Branch 3**: Continues through `conv3`, `ReLU`, `dropout`, and `pool`, then feeds the result into `sogc3`.
3. **Fusion and Classification**:
   - Concatenates the graph-convolved features from the three scales.
   - Flattens the representation and applies a dropout layer.
   - Passes the result through the final linear classifier to obtain logits for the 4 emotion classes.
