# 모델 아키텍처 (`models.py`)

이 프로젝트는 Li 등(2021)이 제안한 **자기 조직화 그래프 신경망(Self-Organized Graph Neural Network, SOGNN)**을 전극 절제 연구에 맞게 조정하여 사용합니다.

## SOGC: 자기 조직화 그래프 합성곱 (Self-Organized Graph Convolution)

SOGNN의 핵심은 `SOGC` 레이어로, 고정된 거리 기반 인접 행렬을 사용하는 대신 모든 입력 샘플에 대해 동적인 그래프 구조를 학습합니다.

```python
class SOGC(nn.Module):
    def __init__(self, n_electrodes, in_features, bn_features, out_features, top_k):
        super().__init__()
        self.n_electrodes = n_electrodes
        self.top_k = min(top_k, n_electrodes)
        self.bn = nn.Linear(in_features, bn_features)
        self.gc = nn.Linear(in_features, out_features)

    def forward(self, x):
        B_E = x.shape[0]
        E = self.n_electrodes
        B = B_E // E

        # [Line 1] 공간 차원 평탄화: (B*E, C, H, W) -> (B, E, C*H*W)
        h = x.reshape(B, E, -1)

        # [Line 2] 인접 행렬 계산: bottleneck -> tanh -> softmax -> top-k
        g = torch.tanh(self.bn(h))                                # (B, E, bn)
        a = torch.softmax(g @ g.transpose(-1, -2), dim=-1)       # (B, E, E)

        # [Line 3] Top-k 희소화 (Sparsification)
        vals, idxs = a.topk(self.top_k, dim=-1)                  # (B, E, k)
        a_sparse = torch.zeros_like(a).scatter_(-1, idxs, vals)   # (B, E, E)

        # [Line 4] 셀프 루프 추가: 대각 성분을 1.0으로 설정
        eye = torch.eye(E, device=a.device, dtype=a.dtype).unsqueeze(0)
        a_sparse = a_sparse * (1 - eye) + eye

        # [Line 5] 그래프 컨볼루션
        out = torch.relu(self.gc(a_sparse @ h))                   # (B, E, out)
        return out
```

### 라인별 상세 설명 (SOGC)
- **Line 1 (`h = x.reshape(...)`)**: 공간적/스펙트럼 특징(채널 × 높이 × 너비)을 각 전극당 하나의 벡터로 평탄화합니다. 이는 각 전극이 노드가 되는 그래프 수준의 연산을 준비하는 단계입니다.
- **Line 2 (`g = ...`, `a = ...`)**: `bn`은 특징을 저차원 병목 공간으로 투영하여 관계를 더 효율적으로 찾습니다. `g @ g.transpose`는 모든 전극 쌍 사이의 유사도를 계산하며, `softmax`는 한 전극에서 다른 전극으로 가는 가중치의 합이 1이 되도록 정규화하여 밀집 인접 행렬을 형성합니다.
- **Line 3 (`a_sparse = ...`)**: 가장 강력한 `top_k` 연결만 유지합니다. `scatter_`는 희소 인접 행렬을 생성하기 위해 top-k 인덱스에만 원래의 softmax 값을 배치하고 나머지는 0으로 채웁니다.
- **Line 4 (`a_sparse * (1 - eye) + eye`)**: 인접 행렬의 대각 성분을 정확히 1.0으로 설정합니다. 이는 메시지 패싱 과정에서 전극 자신의 특징 정보가 항상 보존되도록 보장합니다.
- **Line 5 (`torch.relu(self.gc(a_sparse @ h))`)**: 그래프 컨볼루션을 수행합니다. `a_sparse @ h`는 인접한 전극들로부터 특징을 집계하며, `self.gc` (선형 레이어)는 학습 가능한 변환을 적용합니다.

## SOGNN: 시스템 통합

`SOGNN` 모델은 2D CNN 백본과 여러 `SOGC` 분기를 결합하여 다중 스케일의 공간 및 시간 특징을 캡처합니다.

```python
    def forward(self, x):
        B, E = x.shape[0], x.shape[1]

        # [Line 6] 전극별로 변형: (B*E, 1, bands, T)
        x = x.reshape(B * E, 1, x.shape[2], x.shape[3])

        # [Line 7] 블록 1 + SOGC1 분기
        x = self.pool(self.drop(F.relu(self.conv1(x))))
        x1 = self.sogc1(x)                              # (B, E, 32)

        # [Line 8] 다중 스케일 결합 + 분류
        out = torch.cat([x1, x2, x3], dim=-1)           # (B, E, 96)
        out = self.drop(out)
        out = out.reshape(B, -1)                         # (B, E * 96)
        logits = self.classifier(out)                    # (B, n_classes)
        return logits
```

### 라인별 상세 설명 (SOGNN)
- **Line 6 (`x.reshape(...)`)**: 배치 차원과 전극 차원을 하나로 합칩니다. 이를 통해 2D CNN이 모든 트라이얼의 모든 전극을 독립적인 샘플로 처리하게 하여, CNN이 전극 간의 관계가 아닌 전극 *내부*의 스펙트럼-시간 패턴만을 학습하도록 합니다.
- **Line 7 (`x1 = self.sogc1(x)`)**: CNN 처리 후 특징들이 SOGC 레이어로 전달됩니다. SOGC는 내부적으로 `(B, E)` 구조를 복원하여 처리된 전극 특징들 사이의 관계를 학습합니다.
- **Line 8 (`torch.cat(...)`, `out.reshape(...)`)**: CNN의 저수준, 중간 수준, 고수준 특징들을 결합합니다. 마지막 `reshape(B, -1)`은 모든 전극과 그들의 다중 스케일 특징들을 하나의 긴 벡터로 평탄화하여 최종 분류 결정을 내릴 준비를 합니다.
