# 파이프라인 실행 및 로직 (`main.py`)

이 문서는 데이터 처리, 중요도 지표 및 통계적 검증을 포함한 SOGNN 전극 절제 파이프라인의 6단계 실행 흐름을 설명합니다.

## Phase 0: 데이터 로딩 및 전처리

`load_seed4_session` 함수가 데이터 파이프라인을 구현합니다.

```python
def load_seed4_session(mat_path, session_idx):
    data = sio.loadmat(mat_path)
    labels = SESSION_LABELS[session_idx]
    # ... 트라이얼 추출 ...
    
    # 패딩 전 모든 프레임에서 z-score 통계 계산
    all_frames = np.concatenate(trials_raw, axis=0)  # [Line 9]
    mean = all_frames.mean(axis=0, keepdims=True)    # [Line 10]
    std = all_frames.std(axis=0, keepdims=True) + 1e-8
    
    # 각 트라이얼 정규화, 제로 패딩 및 (62, 5, T_FIXED)로 전치
    X_list = []
    for trial in trials_raw:
        trial_normed = (trial - mean) / std          # [Line 11]
        # ... T_FIXED로 패딩/절단 ...
        X_list.append(trial_padded.transpose(1, 2, 0))
    # ...
```

### 라인별 상세 설명 (전처리)
- **Line 9 (`np.concatenate(trials_raw, axis=0)`)**: 서로 다른 시간 길이를 가진 모든 트라이얼들을 하나의 거대한 배열로 합칩니다. 이는 전체 세션에 대한 글로벌 통계량을 계산하기 위해 필요합니다.
- **Line 10 (`all_frames.mean(...)`)**: 전체 세션 동안 각 (채널, 대역) 쌍에 대한 평균 차분 엔트로피(DE) 값을 계산합니다.
- **Line 11 (`(trial - mean) / std`)**: 실제 Z-score 정규화를 수행합니다. 세션 전체의 평균과 표준편차를 사용함으로써, 세션 간의 편차는 제거하면서 동일 세션 내의 서로 다른 트라이얼 간의 상대적인 강도 차이는 보존합니다.

## Phase 1 & 2: 훈련 및 앙상블

파이프라인은 안정성을 보장하기 위해 여러 무작위 시드(seed)에 대해 반복되는 **Leave-One-Subject-Out (LOSO)** 교차 검증을 사용합니다.

```python
def train_and_evaluate(data, model_cls, model_kwargs, train_kwargs, device='cuda'):
    # ...
    for test_subj in subj_bar:
        # [Line 12] 훈련을 위해 다른 모든 피험자의 모든 세션을 통합
        X_train = np.concatenate([data[s][sess][0] ... if s != test_subj ...])
        # [Line 13] 테스트를 위해 제외된 피험자의 모든 세션을 통합
        X_test = np.concatenate([data[test_subj][sess][0] ...])
        # ... 훈련 루프 ...
```

### 라인별 상세 설명 (LOSO)
- **Line 12 (`X_train = ...`)**: 15명 중 14명의 데이터를 모아 훈련 세트를 구성합니다. 이는 모델이 완전히 새로운 사람(피험자 독립 과제)에 대해 얼마나 잘 일반화되는지 테스트하기 위함입니다.
- **Line 13 (`X_test = ...`)**: 15번째 피험자의 데이터는 오직 평가를 위해서만 엄격하게 분리하여 유지합니다.

## Phase 3: 전극 중요도 지표

중요도는 **치환 중요도 (Permutation Importance, PI)**를 사용하여 계산됩니다.

```python
def permutation_importance(model, X, y, device, n_repeats=10):
    # ...
    for ch in range(n_channels):
        # [Line 14] 배치 전체에서 채널 'ch'의 데이터를 섞음 (shuffle)
        X_perm[:, ch, :, :] = X_perm[perm_idx, ch, :, :]
        # [Line 15] 정확도 하락 측정
        # ...
        acc_importances[ch] = baseline_acc - float(np.mean(shuffled_accs))
```

### 라인별 상세 설명 (중요도)
- **Line 14 (`X_perm[:, ch, ...] = ...`)**: 절제 로직의 핵심입니다. 특정 채널의 데이터를 샘플 간에 무작위로 섞음으로써, 해당 전극의 신호와 실제 감정 레이블 사이의 관계를 파괴합니다. 이때 신호 자체의 분포(평균/분산)는 그대로 유지됩니다.
- **Line 15 (`baseline_acc - ...`)**: 데이터를 섞은 후 정확도가 크게 떨어진다면, 이는 모델이 결정을 내릴 때 해당 전극에 크게 의존하고 있었음을 증명합니다.

## Phase 4: 절제 연구 (Ablation Studies)

절제는 훈련된 모델의 입력에 이진 마스크(binary mask)를 적용하여 수행됩니다.

```python
def evaluate(model, loader, device, channel_mask=None):
    # ...
    if channel_mask is not None:
        mask = channel_mask.to(device).unsqueeze(-1).unsqueeze(-1)  # (1,C,1,1)
        X_batch = X_batch * mask
    logits = model(X_batch)
    # ...
```

- **점진적 절제 (Progressive Ablation)**: "Grand Ranking" (모든 피험자 및 시드에 대한 평균 PI)을 기반으로 전극을 하나씩 제거합니다. 이를 통해 정확도 곡선이 생성됩니다.
- **무릎 지점 탐지 (Knee Point Detection)**: 절제 곡선에서 최대 곡률 지점(무릎 지점)을 찾아 "최적"의 전극 수를 결정합니다.

## Phase 5: 시각화 및 통계

결과는 `matplotlib` 및 `mne`를 사용하여 여러 플롯으로 합성됩니다.

- **토포그래픽 맵 (Topographic Maps)**: `mne.viz.plot_topomap`을 사용하여 중요도 점수를 2D 두피 표현에 투영합니다.
- **통계 검정**: `scipy.stats.wilcoxon`을 사용하여 서로 다른 구성(예: 전체 62채널 vs 10-20 시스템)을 비교하고, 가족오류율(family-wise error rate)을 제어하기 위해 **Holm-Bonferroni 보정**을 적용합니다.

```python
# Holm-Bonferroni 보정 스니펫
n_tests = len(raw_p_values)
sorted_idx = np.argsort(raw_p_values)
for rank, idx in enumerate(sorted_idx):
    adjusted_p[idx] = raw_p_values[idx] * (n_tests - rank)
# ... 단조성 유지 ...
```
