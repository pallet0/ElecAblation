# 설정 및 매핑 (`config.py`)

이 문서는 SOGNN 파이프라인 전반에서 일관성을 유지하기 위해 사용되는 상수, 전극 레이아웃 및 지역별 그룹화에 대해 상세히 설명합니다.

## 데이터셋 상수 (SEED-IV)

| 상수 | 값 | 설명 |
| :--- | :--- | :--- |
| `N_CHANNELS` | 62 | 총 EEG 채널 수 |
| `N_BANDS` | 5 | 주파수 대역: Delta, Theta, Alpha, Beta, Gamma |
| `N_CLASSES` | 4 | 감정 클래스: 중립 (0), 슬픔 (1), 공포 (2), 행복 (3) |
| `N_SUBJECTS` | 15 | 데이터셋의 총 피험자 수 |
| `N_SESSIONS` | 3 | 피험자당 독립적인 기록 세션 수 |
| `T_FIXED` | 64 | SOGNN 입력을 위한 시간적 패딩 길이 (64 프레임) |

## 전극 레이아웃 (Electrode Layout)

`CHANNEL_NAMES`는 SEED-IV `.mat` 파일에 제공된 62채널 시퀀스를 정의합니다. 이 인덱스는 파이프라인 전반에서 마스킹 및 중요도 순위 산정에 사용됩니다.

```python
CHANNEL_NAMES = [
    'FP1','FPZ','FP2','AF3','AF4',                          # 0-4
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',           # 5-13
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',  # 14-22
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',           # 23-31
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',  # 32-40
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',           # 41-49
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',              # 50-56
    'CB1','O1','OZ','O2','CB2'                               # 57-61
]
```

## 지역 및 해부학적 그룹화

이 그룹화는 4단계(절제 실험)에서 특정 뇌 영역을 제거하거나 유지했을 때의 영향을 평가하는 데 사용됩니다.

### 1. 세분화된 지역 (`REGIONS_FINE`)
전방에서 후방으로 이어지는 8개의 겹치지 않는 세로 스트립:
- `prefrontal` (전전두엽), `frontal` (전두엽), `frontal_central` (전두-중앙), `central` (중앙), `central_parietal` (중앙-두정), `parietal` (두정엽), `parietal_occipital` (두정-후두), `occipital` (후두엽).

### 2. 뇌 반구 (`HEMISPHERES`)
- `left` (좌반구): 27개 채널
- `midline` (중앙선): 8개 채널
- `right` (우반구): 27개 채널

### 3. 해부학적 뇌엽 (`LOBES`)
표준 뇌 해부학에 기반한 그룹화:
- `frontal` (전두엽): Fp + AF + F 전극
- `temporal` (측두엽): FT7/8, T7/8, TP7/8
- `central` (중앙): FC, C, CP 전극
- `parietal` (두정엽): P 전극
- `occipital` (후두엽): PO + CB + O 전극

## 표준 몽타주 서브셋 (Montage Subsets)

비교를 위해 사용되는 상용 및 표준 전극 구성:
- `STANDARD_1020`: 국제 10-20 시스템 (19채널)
- `EMOTIV_EPOC`: Emotiv 헤드셋에서 사용하는 14채널 레이아웃
- `MUSE_APPROX`: Muse 헤드밴드를 근사한 4채널 레이아웃

## 시각화 메타데이터

- `MNE_NAME_MAP`: SEED-IV 이름(예: `FPZ`)을 표준 MNE 호환 이름(예: `Fpz`)으로 매핑하는 딕셔너리 (정확한 토포그래픽 플로팅을 위해 사용).
- `SESSION_LABELS`: SEED-IV ReadMe에 정의된 3개 세션, 24개 트라이얼별 감정 정답 레이블.
