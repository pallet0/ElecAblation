# Models Explanation (`models.py`)

This document explains the deep learning models available in the EEG ablation study. The goal is to classify emotions from EEG signals.

**Note:** The current active pipeline (`main.py`) uses the **MLPBaseline** model and extracts channel importance via **Permutation Importance** and **Integrated Gradients**. The Attention-based models described below are available architectures that can be swapped in for future experiments.

## 1. `ChannelAttentionEEGNet`

This model uses an attention mechanism to "learn" which channels are important.

### The Concept: What is Attention?
Imagine you are listening to a crowded room. To understand a specific conversation, you "tune out" the background noise and "focus" on one person's voice.
*   **Without Attention:** The model takes all 62 brain channels, averages them together, and tries to guess the emotion. This is noisy because irrelevant channels confuse the model.
*   **With Attention:** The model looks at each channel and assigns it a "score" (importance). If Channel 1 has a score of 0.9 and Channel 2 has 0.1, the model mostly listens to Channel 1.

### Code Breakdown

#### `__init__` (Setting up the layers)

```python
class ChannelAttentionEEGNet(nn.Module):
    def __init__(self, n_bands=5, n_classes=4, d_hidden=64, dropout=0.5):
        super().__init__()
```
*   **Input:** The model receives data shaped `(Batch_Size, 62_Channels, 5_Frequency_Bands)`.
    *   *Analogy:* For each of the 62 microphones (channels), we have volume levels for Bass, Low-Mid, Mid, High-Mid, High (5 bands).

```python
        self.spectral_encoder = nn.Sequential(
            nn.Linear(n_bands, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
```
*   **Spectral Encoder:** This processes the raw frequency data for *each channel independently*.
    *   It turns the 5 simple numbers (frequency bands) into a richer list of 64 numbers (`d_hidden`).
    *   `GELU`: An activation function (like a neuron firing).
    *   **Result:** Now we have a rich "fingerprint" for what each channel is doing.

```python
        self.attn_scorer = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.Tanh(),
            nn.Linear(d_hidden // 2, 1, bias=False),
        )
```
*   **Attention Scorer (The Judge):** This small network looks at the "fingerprint" of a channel and decides how important it is.
    *   It outputs a single number (score) for each channel.
    *   If the fingerprint looks like "Sadness", it might give a high score. If it looks like "Random Noise", it gives a low score.

```python
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )
```
*   **Classifier:** Once the model has the "focused" brain signal (the weighted sum), this part makes the final decision: "Is it Happy, Sad, Fear, or Neutral?"

#### `forward` (The logic flow)

```python
    def forward(self, x, channel_mask=None):
        h = self.spectral_encoder(x)               # (Batch, 62, 64)
```
*   **Step 1:** Process every channel to get its feature representation (`h`).

```python
        e = self.attn_scorer(h).squeeze(-1)         # (Batch, 62)
```
*   **Step 2:** Calculate raw scores (`e`) for all 62 channels. High number = important.

```python
        if channel_mask is not None:
            e = e.masked_fill(channel_mask == 0, float('-inf'))
```
*   **Step 3: Masking (The Ablation Logic):**
    *   This is critical for our experiment. If we want to simulate "removing" a part of the brain, we don't need to cut the data.
    *   We simply force the attention score to **Negative Infinity** (`-inf`).
    *   When we calculate probabilities next, `softmax(-inf)` becomes exactly **0**.
    *   Effectively, the model ignores that channel completely.

```python
        alpha = F.softmax(e, dim=-1)                # (Batch, 62)
```
*   **Step 4:** Normalize scores (`alpha`). The `softmax` function ensures all channel scores sum up to 1.0 (100%).
    *   Example: `[0.1, 0.8, 0.1]` (Channel 2 is doing 80% of the work).

```python
        context = torch.einsum('bc,bcd->bd', alpha, h)
```
*   **Step 5: Weighted Sum (The Context):**
    *   This combines all channels into **one** global brain representation (`context`).
    *   It multiplies each channel's data (`h`) by its importance (`alpha`).
    *   `0.1*Chan1 + 0.8*Chan2 + 0.1*Chan3`.
    *   The result is a clean signal dominated by the important channels.

```python
        logits = self.classifier(context)
        return logits, alpha
```
*   **Step 6:** Classify the clean signal and return the prediction (`logits`) AND the attention weights (`alpha`) so we can analyze them later.

---

## 2. `MLPBaseline`

This is the primary model used in `main.py` for the ablation study.

```python
class MLPBaseline(nn.Module):
    def __init__(self, input_dim=310, ...):
        # input_dim = 62 channels * 5 bands = 310 numbers
```
*   **Logic:** It takes all 310 numbers, flattens them into a single long list, and feeds them into a standard Neural Network (Multi-Layer Perceptron).
*   **Why use this?** By using a standard model without intrinsic attention, we can apply model-agnostic interpretability methods like **Permutation Importance** and **Integrated Gradients** to objectively measure which channels contribute most to the accuracy. This avoids relying on the model's internal "attention weights" which can sometimes be misleading.

---

## 3. `DualAttentionEEGNet` (Advanced)

This model adds a second layer of attention.

*   **Band Attention:** "Is the Alpha wave more important than the Beta wave right now?"
*   **Channel Attention:** "Is the Frontal Lobe more important than the Temporal Lobe?"

```python
        beta = F.softmax(e_band, dim=-1)                 # (Batch, 62, 5)
        h_chan = torch.einsum('bcn,bcnd->bcd', beta, h_band)
```
*   First, it computes `beta` (importance of frequency bands). It collapses the 5 bands into 1 representation per channel.

```python
        alpha = F.softmax(e_chan, dim=-1)                # (Batch, 62)
        context = torch.einsum('bc,bcd->bd', alpha, h_chan)
```
*   Then, it computes `alpha` (importance of channels) exactly like the first model.

This is a "Hierarchical" approach: Filter frequencies first, then filter locations.
