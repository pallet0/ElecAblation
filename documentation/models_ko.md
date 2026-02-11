# models.md 문서화

이 파일은 EEG 감정 분류를 위해 사용되는 딥 뉴럴 네트워크(DNN) 아키텍처, 특히 자기 조직화 그래프 신경망(Self-Organized Graph Neural Network, SOGNN)을 정의합니다.

## `SOGC` (Self-Organized Graph Convolution) 클래스

`SOGC` 레이어는 고정된 그래프 구조를 사용하는 대신 노드 특징으로부터 희소 인접 행렬(sparse adjacency matrix)을 동적으로 학습합니다.

### 초기화 (`__init__`)
- `n_electrodes`: 그래프의 노드(전극) 수.
- `in_features`: 각 노드의 입력 특징 차원.
- `bn_features`: 인접 행렬 계산에 사용되는 병목(bottleneck) 레이어의 차원.
- `out_features`: 그래프 컨볼루션 후의 출력 특징 차원.
- `top_k`: 각 노드에 대해 유지할 이웃의 수 (희소성 강제).
- `self.bn`: 유사도 계산을 위해 노드 특징을 병목 공간으로 투영하는 선형 레이어.
- `self.gc`: 그래프 컨볼루션(메시지 패싱)을 수행하는 선형 레이어.

### 순전파 (`forward`)
1. **특징 재구성(Reshaping)**: 입력 `x` (B*E, C, H, W)를 (B, E, Features) 형태로 변환합니다.
2. **인접 행렬 계산**:
   - 특징을 병목 레이어에 통과시킨 후 `tanh`를 적용합니다.
   - 내적 유사도(`g @ g^T`)를 사용하여 자기 주의(self-attention)와 유사한 행렬을 계산합니다.
   - 연결을 정규화하기 위해 `softmax`를 적용합니다.
3. **희소화(Sparsification)**: `topk`와 `scatter_`를 사용하여 각 노드에 대해 가장 강한 `top_k`개의 연결만 남깁니다.
4. **자가 루프(Self-loops)**: 인접 행렬의 대각 성분을 1.0으로 설정하여 자가 루프를 추가합니다.
5. **그래프 컨볼루션**: $Y = 	ext{ReLU}(A \cdot X \cdot W)$ 연산을 수행합니다. 여기서 $A$는 학습된 인접 행렬입니다.

---

## `SOGNN` (Self-Organized Graph Neural Network) 클래스

멀티 스케일 CNN 브랜치와 SOGC 레이어를 결합한 메인 모델 아키텍처입니다.

### 초기화 (`__init__`)
- **CNN 블록**: 서로 다른 시간적/스펙트럼 스케일에서 특징을 추출하기 위해 서로 다른 커널 크기를 가진 세 개의 컨볼루션 레이어(`conv1`, `conv2`, `conv3`)를 사용합니다.
- **동적 차원 계산**: 더미 순전파(dummy forward pass)를 사용하여 후속 SOGC 레이어의 입력 크기를 결정합니다.
- **SOGC 브랜치**: CNN 백본의 서로 다른 깊이에서 나온 특징을 처리하는 세 개의 병렬 SOGC 레이어(`sogc1`, `sogc2`, `sogc3`)를 가집니다.
- **분류기(Classifier)**: 세 개의 SOGC 브랜치 출력을 결합하여 클래스 로짓(logits)을 생성하는 최종 선형 레이어입니다.

### 순전파 (`forward`)
1. **입력 재구성**: 입력 EEG 데이터 (Batch, Electrodes, Bands, Time)를 CNN 단계에서 처리하기 위해 전극별 포맷으로 변환합니다.
2. **멀티 스케일 특징 추출**:
   - **브랜치 1**: 데이터를 `conv1`, `ReLU`, `dropout`, `pool`에 통과시킨 후 그 결과를 `sogc1`에 입력합니다.
   - **브랜치 2**: 이전 CNN 출력을 `conv2`, `ReLU`, `dropout`, `pool`에 이어서 통과시킨 후 `sogc2`에 입력합니다.
   - **브랜치 3**: `conv3`, `ReLU`, `dropout`, `pool`을 거쳐 `sogc3`에 입력합니다.
3. **융합 및 분류**:
   - 세 가지 스케일에서 나온 그래프 컨볼루션 특징들을 결합(concatenate)합니다.
   - 표현을 평탄화(flatten)하고 드롭아웃 레이어를 적용합니다.
   - 최종 선형 분류기를 통과시켜 4개 감정 클래스에 대한 로짓을 얻습니다.
