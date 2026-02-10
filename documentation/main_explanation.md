# Main Pipeline Explanation (`main.py`)

This file runs the entire scientific experiment. It connects data loading, model training, and the "ablation study" (systematically disabling brain regions to see what happens).

## 1. Data Loading

### `load_seed4_session(mat_path, session_idx)`
*   **Goal:** Read the raw `.mat` files provided by the dataset.
*   **Input:** A file containing EEG features (Differential Entropy) for one subject in one session.
*   **Process:**
    1.  It loops through 24 trials (movie clips that induce emotions).
    2.  `trial_data` originally has shape `(62 channels, Time, 5 bands)`.
    3.  It transposes it to `(Time, 62, 5)` because PyTorch models usually expect the batch (Time) dimension first.
    4.  It creates a label list `y` (0=Neutral, 1=Sad, 2=Fear, 3=Happy) corresponding to the movie clip shown.
*   **Output:** A massive array `X` of all EEG samples and `y` labels.

## 2. The Ablation Tool

### `make_channel_mask(active_indices, ...)`
*   **Concept:** To simulate "turning off" a part of the brain, we create a **Mask**.
*   **Logic:**
    *   Create a row of 62 zeros: `[0, 0, 0, ... 0]`.
    *   Set the `active_indices` (the channels we want to KEEP) to 1: `[0, 1, 1, ... 0]`.
*   **Usage:** This mask is sent to the model. As explained in `models_explanation.md`, the model replaces the 0s with `-infinity`, effectively silencing those channels.

## 3. Training Helpers

### `train_one_epoch(...)` & `evaluate(...)`
*   Standard Deep Learning loops.
*   **Key Detail:** `evaluate` accepts a `channel_mask`.
    *   When training, we usually use no mask (mask=None), letting the model learn from the whole brain.
    *   When testing specifically for ablation, we pass the mask to see how the model fails without certain channels.

## 4. The Experiment Phases

The script runs in 5 distinct phases.

### Phase 0: Data Preparation
*   Loads all 15 subjects.
*   **Z-Score Normalization:**
    *   `X = (X - mean) / std`
    *   This scales all data to be roughly between -1 and 1. This is crucial for neural networks to learn stably.

### Phase 1: Hyperparameter Search (Cross-Validation)
*   **Goal:** Find the best settings (Learning Rate, Model Size, etc.) without cheating by looking at the test set.
*   **Method:**
    *   It pools data from Sessions 1 and 2.
    *   It uses **5-Fold Cross-Validation**: It splits the data into 5 parts. It trains on 4 and tests on 1, rotating 5 times.
    *   This gives a robust estimate of "how good this configuration is generally."

### Phase 2: Per-Subject Training
*   **Goal:** Train a personalized model for each of the 15 subjects.
*   **Data Split:**
    *   **Train:** Session 1 + Session 2.
    *   **Test:** Session 3.
*   **Why?** EEG varies wildly between people. A generic "one-size-fits-all" model often fails. We train 15 separate models.
*   **Early Stopping:** If the model stops improving on the test set, we stop training to prevent "overfitting" (memorizing noise).

### Phase 3: Extracting Importance
*   **Goal:** Ask the trained models: "Which channels did you find useful?"
*   **Method:**
    *   Run the test data through the model.
    *   Collect the `alpha` (attention weights) from the model.
    *   Average them over time.
    *   **Grand Ranking:** Average the importance across all 15 subjects to find the universally important brain areas.

### Phase 4: Full Ablation Study (The Core Experiment)
Now we test the scientific hypotheses using `run_full_ablation_study`.

#### A. Region Ablation
*   **Question:** "Is the Frontal Lobe necessary?"
*   **Action:**
    1.  **Keep Only:** Mask everything *except* the Frontal channels. Test accuracy.
    2.  **Remove:** Mask *only* the Frontal channels. Test accuracy.
*   We do this for Frontal, Temporal, Parietal, Occipital, etc.

#### B. Hemisphere Asymmetry
*   **Question:** "Is the Left brain more emotional than the Right?"
*   **Action:** Keep only Left channels vs Keep only Right channels.

#### C. Montage Analysis (Commercial Headsets)
*   **Question:** "Could we detect emotions with a cheap 4-channel headset (Muse)?"
*   **Action:** Mask all channels except the 4 used by the Muse headband (TP9, AF7, AF8, TP10). Compare accuracy to the full 62-channel cap.

#### D. Progressive Ablation (The "Curve")
*   **Question:** "How many channels do we *really* need? 10? 20?"
*   **Strategy 1 (Smart):** Keep the top 5 most important channels (from Phase 3), then top 10, top 15...
*   **Strategy 2 (Random):** Keep 5 random channels, 10 random...
*   **Expectation:** The "Smart" curve should rise much faster than the "Random" curve. This proves our Attention mechanism actually found meaningful signals.

### Phase 5: Visualization
*   Generates PDFs to visualize the results:
    *   `topomap_attention.pdf`: A heatmap of the brain showing hot spots.
    *   `progressive_ablation.pdf`: A line graph showing accuracy vs number of channels.
    *   `region_ablation.pdf`: Bar charts comparing brain regions.

## 5. Statistical Tests
*   Uses **Wilcoxon Signed-Rank Test**.
*   **Why?** To prove the results aren't just luck.
*   It compares the list of 15 subject accuracies for "Full Cap" vs "Muse Headset".
*   If p < 0.05, the difference is statistically significant.
