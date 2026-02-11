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

## 3. Experimental Architecture: MLPBaseline
The study employs a standard Multi-Layer Perceptron (MLP) as a robust baseline for emotion classification. This approach allows for model-agnostic feature importance analysis.
1.  **Input:** Flattened vector of 310 features (62 channels × 5 bands).
2.  **Hidden Layers:** Two fully connected layers with BatchNorm, ReLU activation, and Dropout.
3.  **Classifier:** A final linear layer to predict the 4 emotion classes.

## 4. Experimental Pipeline

### Phase 1: Training Strategy
*   **Hyperparameter Search:** 5-fold Stratified Cross-Validation on pooled data from Sessions 1 and 2 to find optimal dropout and learning rates.
*   **Multi-Seed Ensemble:** Each subject is trained across multiple random seeds to ensure the results are statistically stable and robust against initialization noise.
*   **Per-Subject Training:** Each subject gets a personalized model.
    *   **Training Set:** Sessions 1 & 2.
    *   **Test Set:** Session 3 (to evaluate cross-day generalization).
    *   **Regularization:** Early stopping is used to prevent the model from memorizing the specific session noise.

### Phase 2: Channel Importance Extraction
Once the models are trained, we identify critical electrodes using two post-hoc interpretability methods:
1.  **Permutation Importance (PI):** We systematically shuffle the data of one channel at a time in the test set and measure the drop in accuracy. Larger drops indicate higher importance.
2.  **Integrated Gradients (IG):** We compute the gradient of the prediction with respect to the input features, providing a fine-grained attribution map.
Importance scores are averaged across all subjects and random seeds to generate a **Grand Ranking**.
*   **Stability Analysis:** Spearman rank correlation is calculated between different seeds to verify the consistency of the identified "hot spots."

### Phase 3: Systematic Ablation Study
We "ablate" (disable) sensors to test the following hypotheses:
1.  **Regional Necessity:** Which lobes (Frontal, Temporal, etc.) can be removed with the least impact on accuracy?
2.  **Hemisphere Asymmetry:** Does the model perform better using only the left or right hemisphere?
3.  **Montage Practicality:** How does a research-grade 62-channel setup compare to standard 10-20 (19 ch), Emotiv EPOC (14 ch), and Muse (4 ch) configurations?
4.  **Progressive Thinning:** We compare **Importance-guided removal** (removing least important first based on PI ranking) against **Random removal**. All results are averaged across the ensemble of models to ensure reliability.

## 5. Statistical Validation
To ensure findings are robust, results are analyzed using the **Wilcoxon Signed-Rank Test**. Because multiple comparisons are made (e.g., comparing several hardware montages), the **Holm-Bonferroni Correction** is applied to prevent "Type I" errors (false positives), ensuring that any reported performance difference is statistically sound.

## 6. Expected Deliverables
*   **Topographic Heatmaps:** Visualizing where the "emotional brain" is most active (derived from PI and IG).
*   **Ablation Curves:** Quantifying the trade-off between the number of sensors and system accuracy.
*   **Hardware Recommendations:** Determining the optimal minimal electrode set for portable emotion-sensing headsets.
