# config.py 문서화

이 파일은 SEED-IV EEG 감정 인식 절제 연구(ablation study)에 사용되는 상수, 전극 매핑 및 지역 그룹화를 포함합니다.

## EEG 채널 레이아웃
- `CHANNEL_NAMES`: SEED-IV 레이아웃을 따르는 62개 EEG 전극 이름 리스트.
- `N_CHANNELS`: 총 EEG 채널 수 (62개).
- `N_BANDS`: 미분 엔트로피(Differential Entropy, DE) 특징에 사용되는 주파수 대역 수 (5개: delta, theta, alpha, beta, gamma).
- `N_CLASSES`: 감정 클래스 수 (4개: neutral, sad, fear, happy).
- `N_SUBJECTS`: SEED-IV 데이터셋의 피험자 수 (15명).
- `N_SESSIONS`: 피험자당 세션 수 (3회).
- `T_FIXED`: SOGNN 모델 입력을 위한 시간적 패딩(temporal padding) 길이 (64).

## 데이터 설정
- `DATA_ROOT`: 전처리된 SEED-IV 특징(`eeg_feature_smooth`)이 저장된 디렉토리 경로.
- `SESSION_LABELS`: 각 3개 세션의 24개 트라이얼(trial)에 대한 정답 라벨을 포함하는 중첩 리스트.

## 지역 그룹화 (전극 서브셋)
이 그룹화는 감정 인식에서 서로 다른 뇌 영역의 기여도를 분석하는 데 사용됩니다.

### `REGIONS_FINE`
8개의 겹치지 않는 종방향 스트립(전방에서 후방으로)을 해당 채널 인덱스에 매핑한 딕셔너리:
- `prefrontal`, `frontal`, `frontal_central`, `central`, `central_parietal`, `parietal`, `parietal_occipital`, `occipital`.

### `HEMISPHERES`
뇌 반구를 채널 인덱스에 매핑한 딕셔너리:
- `left` (좌반구), `right` (우반구), `midline` (중앙선).

### `LOBES`
해부학적 엽(lobe) 그룹화:
- `frontal` (전두엽), `temporal` (측두엽), `central` (중앙부), `parietal` (두정엽), `occipital` (후두엽).

## 표준 몽타주 서브셋
일반적인 축소 전극 구성을 위한 인덱스:
- `STANDARD_1020`: 국제 10-20 시스템 (19채널).
- `EMOTIV_EPOC`: Emotiv EPOC 헤드셋에서 사용하는 14채널 레이아웃.
- `MUSE_APPROX`: Muse 헤드밴드의 4채널 근사치.

## MNE 매핑
- `MNE_NAME_MAP`: 지형도 시각화(topographic plotting) 라이브러리와의 호환성을 위해 SEED-IV 채널 이름을 표준 MNE 이름으로 매핑합니다.
