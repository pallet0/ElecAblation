# Research Report: EEG Electrode Ablation Study for Emotion Recognition

## 1. Objective
The goal of this study is to identify the most critical EEG electrodes and brain regions for emotion recognition. By using an **Attention-Based Neural Network**, we aim to autonomously rank electrode importance and determine the minimum number of sensors required to maintain high classification accuracy. This has direct implications for the development of wearable Brain-Computer Interfaces (BCI).

## 2. Dataset and Features
*   **Dataset:** SEED-IV (SJTU Emotion EEG Dataset).
    *   **Subjects:** 15 individuals.
    *   **Sessions:** 3 sessions per subject, recorded on different days.
    *   **Emotions:** 4 classes (Neutral, Sad, Fear, Happy).
    *   **Stimuli:** 24 movie clips per session.
*   **Features:** Differential Entropy (DE) features extracted from 62 EEG channels across 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma).
*   **Preprocessing:** Data is z-score normalized per session to handle inter-session and inter-subject variability.

## 3. Proposed Architecture: ChannelAttentionEEGNet
The study employs a custom deep learning model designed to interpret brain topography:
1.  **Spectral Encoder:** A series of linear layers that transform the 5-band frequency data into a high-dimensional representation for each electrode.
2.  **Channel Attention (Bahdanau-style):** A judge network that calculates an importance score ($\alpha$) for each electrode. This allows the model to "focus" on specific brain regions while ignoring noisy or irrelevant sensors.
3.  **Global Integration:** Electrodes are combined via a weighted sum based on their attention scores.
4.  **Classifier:** A final MLP that predicts the emotion from the integrated brain signal.

## 4. Experimental Pipeline

### Phase 1: Training Strategy
*   **Hyperparameter Search:** 5-fold Stratified Cross-Validation on pooled data from Sessions 1 and 2 to find optimal dropout and learning rates.
*   **Per-Subject Training:** Each subject gets a personalized model.
    *   **Training Set:** Sessions 1 & 2.
    *   **Test Set:** Session 3 (to evaluate cross-day generalization).
    *   **Regularization:** Early stopping is used to prevent the model from memorizing the specific session noise.

### Phase 2: Channel Importance Extraction
Once the models are trained, we extract the attention weights ($\alpha$) from the test set. By averaging these weights across all 15 subjects, we generate a **Grand Ranking** of the 62 electrodes, identifying the "hot spots" for emotion processing.

### Phase 3: Systematic Ablation Study
We "ablate" (disable) sensors to test the following hypotheses:
1.  **Regional Necessity:** Which lobes (Frontal, Temporal, etc.) can be removed with the least impact on accuracy?
2.  **Hemisphere Asymmetry:** Does the model perform better using only the left or right hemisphere?
3.  **Montage Practicality:** How does a research-grade 62-channel setup compare to standard 10-20 (19 ch), Emotiv EPOC (14 ch), and Muse (4 ch) configurations?
4.  **Progressive Thinning:** We compare **Attention-guided removal** (removing least important first) against **Random removal**. If the Attention-guided curve is superior, it validates that our model has learned genuine neurophysiological markers.

## 5. Statistical Validation
To ensure findings are robust, results are analyzed using the **Wilcoxon Signed-Rank Test**. Because multiple comparisons are made (e.g., comparing several hardware montages), the **Bonferroni Correction** is applied to prevent "Type I" errors (false positives), ensuring that any reported performance difference is statistically sound.

## 6. Expected Deliverables
*   **Topographic Heatmaps:** Visualizing where the "emotional brain" is most active.
*   **Ablation Curves:** Quantifying the trade-off between the number of sensors and system accuracy.
*   **Hardware Recommendations:** Determining the optimal minimal electrode set for portable emotion-sensing headsets.
