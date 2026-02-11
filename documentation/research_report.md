# Research Report: EEG Electrode Ablation Study for Emotion Recognition

## 1. Objective and Motivation
The primary objective of this study is to perform a rigorous neurophysiological and computational analysis of the SEED-IV dataset to identify the most critical EEG electrodes and brain regions for multi-class emotion recognition. 

While research-grade EEG caps utilize 62 or more channels, such setups are impractical for real-world applications. This study addresses the "Curse of Dimensionality" in EEG data—where the high number of features often leads to overfitting—by identifying a minimal, optimal subset of sensors. By using a model-agnostic interpretability framework, we aim to:
1.  **Map the Emotional Brain:** Autonomously identify "hot spots" for emotion processing (Neutral, Sad, Fear, Happy).
2.  **Validate Interpretability:** Prove that statistical importance (Permutation Importance/Integrated Gradients) aligns with functional brain anatomy.
3.  **Optimize BCI Hardware:** Determine the performance trade-offs for commercial montages (10-20 system, Emotiv EPOC, Muse) to guide the design of next-generation wearable Brain-Computer Interfaces.

## 2. Dataset and Feature Engineering
*   **SEED-IV Dataset:** A benchmark dataset from Shanghai Jiao Tong University.
    *   **Subjects:** 15 healthy subjects (balanced for gender).
    *   **Sessions:** 3 separate sessions per subject, recorded on different days to ensure cross-day variability.
    *   **Stimuli:** 24 movie clips (6 clips per emotion: Neutral, Sad, Fear, Happy) per session.
*   **Differential Entropy (DE) Features:**
    *   Unlike raw PSD, DE has been shown to be more stable and discriminative for emotion recognition.
    *   **Frequency Bands:** Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), and Gamma (30-70Hz).
    *   **Input Dimension:** Each trial is treated as one sample with shape `(62, 5, T)` — 62 channels × 5 bands × T time frames, zero-padded to T=64.
*   **Preprocessing:**
    *   **Z-Score Normalization:** Performed per session on all frames *before* zero-padding (`X = (X - μ) / σ`). This mitigates EEG non-stationarity and ensures the zero-padded region represents true zero (≈ session mean).

## 3. Experimental Architecture: SOGNN
We employ a **Self-Organized Graph Neural Network** (SOGNN; Li et al., 2021), which learns dynamic inter-electrode relationships via self-organized graph convolution. Unlike flat MLP classifiers, SOGNN's graph structure captures spatial dependencies between electrodes — making it well suited for studying which electrodes (graph nodes) drive classification.

*   **Per-electrode feature extraction:** 3 Conv2d+MaxPool blocks process each electrode's `(5, T)` DE features independently, producing a 512-dim node feature vector.
*   **Self-organized graph convolution (×3 layers):** A dynamic adjacency matrix is computed per sample via `A = softmax(tanh(HW) · tanh(HW)^T)`, sparsified to top-k neighbors. Graph convolution: `H' = ReLU(A_sparse · H · W_gc)`. Output: 64-dim per node.
*   **Classification:** Node features are flattened (preserving electrode identity) and passed through a linear layer to 4 classes. Total: ~130K parameters.
*   **Regularization:** Dropout (0.1), weight decay (1e-4), label smoothing (0.1), CosineAnnealingLR, gradient clipping (max_norm=1.0), early stopping on validation accuracy.

**Note on interpretability methods.** Permutation Importance (PI) remains fully model-agnostic — it treats SOGNN as a black box. Integrated Gradients (IG) computes attributions through SOGNN's graph operations. The retrain-from-scratch ablation validates mask-based results by truly removing electrodes from the graph.

**Deviations from the original SOGNN paper.** We use within-subject evaluation (train sessions 1+2, test session 3) rather than cross-subject LOSO, early stopping on validation accuracy rather than a training AUC threshold, and add label smoothing and cosine LR scheduling as regularization for the small per-subject sample size (48 train / 24 test).

## 4. Experimental Pipeline

### Phase 1: Hyperparameter Optimization and Ensemble
*   **HP Search:** We perform a grid search using **5-Fold GroupKFold Cross-Validation** on pooled data from Sessions 1 and 2. Trial IDs are used as groups to prevent temporal leakage (ensuring the model doesn't see the same movie clip in both train and validation sets).
*   **Multi-Seed Ensemble:** Every experiment is repeated across 5 random seeds. Final results (accuracies and importance scores) are averaged across this ensemble to minimize the impact of stochastic weight initialization.

### Phase 2: Dual-Method Importance Extraction
We identify critical electrodes using two complementary axiomatic and statistical methods:
1.  **Permutation Importance (PI):** A global, model-agnostic method. We shuffle the temporal data of channel $i$ and observe the accuracy drop $\Delta Acc$. This captures both linear and non-linear dependencies.
2.  **Integrated Gradients (IG):** An axiomatic attribution method. We calculate the integral of the gradients along the path from a baseline (zero input) to the actual input. This provides fine-grained, sample-level attribution that is then averaged.
*   **Grand Ranking:** A final ranking is derived by averaging scores across all 15 subjects and 5 seeds.
*   **Stability Diagnostics:** We use **Spearman Rho** to measure the correlation of rankings between seeds, ensuring that the identified "emotional hot spots" are consistent and reliable.

### Phase 3: Systematic Ablation Study
We systematically "ablate" (disable) sensors to test specific neuroscientific hypotheses:
1.  **Regional Sensitivity:** Disabling specific regions (e.g., Prefrontal, Frontal, Parietal, Occipital). We test "Keep-Only" (sufficiency) and "Remove-Only" (necessity) scenarios.
2.  **Hemispheric Asymmetry:** Comparing the predictive power of the Left vs. Right hemisphere to investigate lateralization of emotions.
3.  **Montage Practicality:** Evaluating accuracy drops when restricted to hardware like the **Emotiv EPOC (14ch)** or **Muse (4ch)**.
4.  **Progressive Thinning Curves:** We plot accuracy as a function of the number of remaining channels. We compare:
    *   **PI-Guided:** Removing least important channels first.
    *   **Random:** Averaged over 20 random permutations.
    The gap between these curves quantifies the "Information Density" of our identified electrodes.

## 5. Statistical Validation
To distinguish genuine neurophysiological effects from random noise:
*   **Wilcoxon Signed-Rank Test:** A non-parametric test used to compare the paired accuracies of the 15 subjects across different configurations (e.g., Full Cap vs. 10-20 Montage).
*   **Holm-Bonferroni Correction:** Since we perform multiple comparisons, we adjust the p-values to control the Family-Wise Error Rate (FWER), ensuring that a reported significance of $p < 0.05$ is truly meaningful.

## 6. Expected Deliverables and Impact
*   **Topographic Importance Maps:** High-resolution heatmaps showing the dominance of Frontal and Temporal regions in emotion processing.
*   **Ablation Curves:** Quantitative proof of the minimum sensor count (e.g., "12 channels achieve 90% of full-cap performance").
*   **Hardware Design Guidelines:** A recommended "Minimal Emotion Montage" for developers building low-cost, portable BCI systems for mental health monitoring and affective computing.