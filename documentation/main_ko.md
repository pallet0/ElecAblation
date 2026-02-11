# main.py 문서화

이 스크립트는 SEED-IV 데이터셋에서 SOGNN 모델을 사용한 EEG 전극 절제 연구(ablation study)의 전체 파이프라인을 구현합니다.

## 핵심 함수

### 데이터 로딩 및 전처리
- `load_seed4_session(mat_path, session_idx)`:
  - `.mat` 파일에서 미분 엔트로피(DE) 특징을 로드합니다.
  - 데이터를 (Trial, Electrodes, Bands, Time) 형태로 변환합니다.
  - 한 세션 내의 모든 트라이얼에 대해 z-score 정규화 통계치를 계산합니다.
  - 트라이얼을 고정된 길이(`T_FIXED`)로 제로 패딩하거나 자릅니다.
- `make_channel_mask(active_indices, n_channels, batch_size)`:
  - 전극 제거를 시뮬레이션하기 위해 평가 중에 사용되는 이진 마스크(유지할 채널은 1.0, 제거할 채널은 0.0)를 생성합니다.

### 학습 및 평가 헬퍼
- `train_one_epoch(...)`: 한 에포크에 대한 표준 PyTorch 학습 루프입니다.
- `evaluate(...)`: 모델 정확도를 평가합니다. 특정 전극을 0으로 만드는 `channel_mask` 적용을 지원합니다.
- `compute_training_auc(...)`: 중단 기준으로 사용되는 다중 클래스 분류를 위한 macro-averaged AUC를 계산합니다.
- `train_and_evaluate(...)`: **Leave-One-Subject-Out (LOSO)** 교차 검증 스킴을 구현합니다. 14명의 피험자로 학습하고 15번째 피험자로 테스트하는 과정을 모든 폴드에 대해 반복합니다.

### 중요도 계산
- `permutation_importance(model, X, y, device, n_repeats)`:
  - 특정 채널의 데이터를 샘플 간에 섞고 정확도 하락 폭을 측정하여 전극 중요도를 측정합니다.
- `per_emotion_permutation_importance(...)`: 각 감정 클래스별로 순열 중요도(permutation importance)를 계산합니다.
- `integrated_gradients_importance(...)`:
  - Integrated Gradients (IG) 속성(attribution) 방법을 구현합니다.
  - 기준점(0)에서 실제 입력까지의 경로를 따라 그래디언트의 적분을 계산합니다.
  - 시간 및 주파수 대역에 대해 절대 IG 값을 합산하여 채널별 중요도 점수를 얻습니다.

### 절제 연구(Ablation Study) 로직
- `run_full_ablation_study(...)`:
  - 서로 다른 지역, 엽(lobe), 반구에 대해 마스크를 사용하여 사전 학습된 모델을 평가합니다.
  - **점진적 절제(Progressive Ablation)**를 수행합니다: 중요도 순위(가장 중요하지 않은 것부터 또는 가장 중요한 것부터) 또는 무작위로 전극을 점진적으로 제거합니다.
- `run_retrain_ablation_study(...)`:
  - 마스크 기반 절제와 달리, 각 설정에 대해 **모델을 처음부터 다시 학습**합니다 (예: 62채널 모델을 마스킹하는 대신 4채널 모델을 직접 학습).
  - 모델이 축소된 전극 세트에 적응할 수 있는지 확인하는 데 사용됩니다.

### 시각화 및 통계
- `plot_progressive_ablation_curves(...)`: 채널 수에 따른 정확도 변화 곡선을 그립니다.
- `plot_topographic_importance(...)`: `mne`를 사용하여 2D 두피 투영도에 전극 중요도 히트맵을 생성합니다.
- `plot_region_ablation_table(...)` & `plot_lobe_ablation_table(...)`: 지역별/엽별 유지 및 제거 시의 정확도를 비교하는 막대 그래프를 생성합니다.
- `plot_retrain_comparison(...)`: 마스크 기반 절제와 처음부터 다시 학습하는 방식의 정확도를 비교합니다.

## 실행 흐름 (`if __name__ == '__main__':`)

1. **Phase 0 (데이터 로딩)**: 전체 SEED-IV 데이터셋을 메모리에 로드합니다.
2. **다중 시드 앙상블**: 통계적 안정성을 위해 LOSO 파이프라인을 여러 번 실행합니다 (`--n_seeds`로 조절).
3. **Phase 2 (학습)**: 각 LOSO 폴드에 대해 SOGNN 모델을 학습합니다.
4. **Phase 3 (중요도)**: 각 피험자와 전극에 대해 순열 중요도와 Integrated Gradients를 계산합니다.
5. **집계(Aggregation)**: 피험자와 시드 전체에 대해 중요도 점수를 평균하여 "전체 순위(Grand Ranking)"를 생성합니다.
6. **Phase 4 (절제 연구)**: 절제 실험(마스크 기반 및 선택적으로 재학습 기반)을 실행합니다.
7. **Phase 5 (결과)**:
   - 모든 PDF 플롯을 생성합니다.
   - **Holm-Bonferroni 교정**을 적용한 **Wilcoxon 부호 순위 검정**을 통해 통계적 검정을 수행합니다.
   - 모든 수치 결과를 `results.json`에 저장합니다.
