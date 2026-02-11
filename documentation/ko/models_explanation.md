# 모델 설명 (`models.py`)

이 문서는 EEG 제거 연구(Ablation Study)에 사용 가능한 딥러닝 모델들을 설명합니다. 목표는 EEG 신호로부터 감정을 분류하는 것입니다.

**참고:** 현재 활성 파이프라인(`main.py`)은 **MLPBaseline** 모델을 사용하며, **순열 중요도(Permutation Importance)** 및 **통합 기울기(Integrated Gradients)**를 통해 채널 중요도를 추출합니다. 아래 설명된 어텐션 기반 모델들은 향후 실험을 위해 교체하여 사용할 수 있는 아키텍처들입니다.

## 1. `ChannelAttentionEEGNet`

이 모델은 어텐션 메커니즘을 사용하여 어떤 채널이 중요한지 "학습"합니다.

### 개념: 어텐션(Attention)이란 무엇인가?
시끄러운 방에서 듣고 있다고 상상해 보세요. 특정 대화를 이해하기 위해, 당신은 배경 소음을 "무시하고" 한 사람의 목소리에 "집중"합니다.
*   **어텐션 없음:** 모델은 62개의 모든 뇌 채널을 가져와 평균을 내고 감정을 추측하려 합니다. 관련 없는 채널들이 모델을 혼란스럽게 하기 때문에 노이즈가 많습니다.
*   **어텐션 있음:** 모델은 각 채널을 살펴보고 "점수"(중요도)를 부여합니다. 만약 채널 1의 점수가 0.9이고 채널 2가 0.1이라면, 모델은 주로 채널 1의 신호를 듣습니다.

### 코드 분석

#### `__init__` (계층 설정)

```python
class ChannelAttentionEEGNet(nn.Module):
    def __init__(self, n_bands=5, n_classes=4, d_hidden=64, dropout=0.5):
        super().__init__()
```
*   **입력:** 모델은 `(배치 크기, 62 채널, 5 주파수 대역)` 형태의 데이터를 받습니다.
    *   *비유:* 62개의 마이크(채널) 각각에 대해, 우리는 저음, 중저음, 중음, 중고음, 고음(5개 대역)의 볼륨 레벨을 가지고 있습니다.

```python
        self.spectral_encoder = nn.Sequential(
            nn.Linear(n_bands, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
```
*   **스펙트럼 인코더 (Spectral Encoder):** *각 채널을 독립적으로* 처리하여 원본 주파수 데이터를 가공합니다.
    *   5개의 단순한 숫자(주파수 대역)를 64개의 숫자(`d_hidden`)로 이루어진 더 풍부한 리스트로 변환합니다.
    *   `GELU`: 활성화 함수(뉴런이 발화하는 것과 같음).
    *   **결과:** 이제 우리는 각 채널이 무엇을 하고 있는지에 대한 풍부한 "지문(fingerprint)"을 갖게 됩니다.

```python
        self.attn_scorer = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.Tanh(),
            nn.Linear(d_hidden // 2, 1, bias=False),
        )
```
*   **어텐션 채점자 (심판):** 이 작은 네트워크는 채널의 "지문"을 보고 그것이 얼마나 중요한지 결정합니다.
    *   각 채널에 대해 하나의 숫자(점수)를 출력합니다.
    *   지문이 "슬픔"처럼 보이면 높은 점수를 줄 수 있습니다. "무작위 소음"처럼 보이면 낮은 점수를 줍니다.

```python
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )
```
*   **분류기 (Classifier):** 모델이 "집중된" 뇌 신호(가중 합)를 얻으면, 이 부분이 최종 결정을 내립니다: "행복, 슬픔, 공포, 아니면 중립인가?"

#### `forward` (논리 흐름)

```python
    def forward(self, x, channel_mask=None):
        h = self.spectral_encoder(x)               # (배치, 62, 64)
```
*   **1단계:** 모든 채널을 처리하여 특징 표현(`h`)을 얻습니다.

```python
        e = self.attn_scorer(h).squeeze(-1)         # (배치, 62)
```
*   **2단계:** 모든 62개 채널에 대한 원시 점수(`e`)를 계산합니다. 높은 숫자는 중요함을 의미합니다.

```python
        if channel_mask is not None:
            e = e.masked_fill(channel_mask == 0, float('-inf'))
```
*   **3단계: 마스킹 (제거 로직):**
    *   이것은 우리 실험에 매우 중요합니다. 뇌의 일부를 "제거"하는 시뮬레이션을 하려면, 데이터를 자를 필요가 없습니다.
    *   우리는 단순히 어텐션 점수를 **음의 무한대**(`-inf`)로 강제합니다.
    *   다음에 확률을 계산할 때, `softmax(-inf)`는 정확히 **0**이 됩니다.
    *   사실상, 모델은 그 채널을 완전히 무시하게 됩니다.

```python
        alpha = F.softmax(e, dim=-1)                # (배치, 62)
```
*   **4단계:** 점수 정규화(`alpha`). `softmax` 함수는 모든 채널 점수의 합이 1.0(100%)이 되도록 보장합니다.
    *   예: `[0.1, 0.8, 0.1]` (채널 2가 작업의 80%를 수행함).

```python
        context = torch.einsum('bc,bcd->bd', alpha, h)
```
*   **5단계: 가중 합 (컨텍스트):**
    *   이것은 모든 채널을 **하나의** 전역 뇌 표현(`context`)으로 결합합니다.
    *   각 채널의 데이터(`h`)에 중요도(`alpha`)를 곱합니다.
    *   `0.1*채널1 + 0.8*채널2 + 0.1*채널3`.
    *   결과는 중요한 채널들이 지배하는 깨끗한 신호입니다.

```python
        logits = self.classifier(context)
        return logits, alpha
```
*   **6단계:** 깨끗한 신호를 분류하고 예측값(`logits`)과 어텐션 가중치(`alpha`)를 반환하여 나중에 분석할 수 있게 합니다.

---

## 2. `MLPBaseline`

이것은 `main.py`의 제거 연구에서 사용되는 주 모델입니다.

```python
class MLPBaseline(nn.Module):
    def __init__(self, input_dim=310, ...):
        # 입력 차원 = 62 채널 * 5 대역 = 310개의 숫자
```
*   **로직:** 310개의 숫자를 모두 가져와 하나의 긴 리스트로 펼치고(flatten), 표준 신경망(다층 퍼셉트론)에 입력합니다.
*   **왜 이것을 사용하는가?** 내재적인 어텐션 없이 표준 모델을 사용함으로써, **순열 중요도**나 **통합 기울기**와 같은 모델 불가지론적(model-agnostic) 해석 방법을 적용하여 어떤 채널이 정확도에 가장 크게 기여하는지 객관적으로 측정할 수 있습니다. 이는 때때로 오해를 불러일으킬 수 있는 모델 내부의 "어텐션 가중치"에 의존하는 것을 방지합니다.

---

## 3. `DualAttentionEEGNet` (고급)

이 모델은 어텐션의 두 번째 층을 추가합니다.

*   **대역 어텐션 (Band Attention):** "지금 알파파가 베타파보다 더 중요한가?"
*   **채널 어텐션 (Channel Attention):** "전두엽이 측두엽보다 더 중요한가?"

```python
        beta = F.softmax(e_band, dim=-1)                 # (배치, 62, 5)
        h_chan = torch.einsum('bcn,bcnd->bcd', beta, h_band)
```
*   먼저, `beta`(주파수 대역의 중요도)를 계산합니다. 5개의 대역을 채널당 1개의 표현으로 축소합니다.

```python
        alpha = F.softmax(e_chan, dim=-1)                # (배치, 62)
        context = torch.einsum('bc,bcd->bd', alpha, h_chan)
```
*   그 다음, 첫 번째 모델과 정확히 같이 `alpha`(채널의 중요도)를 계산합니다.

이것은 "계층적(Hierarchical)" 접근 방식입니다: 주파수를 먼저 필터링하고, 그 다음 위치를 필터링합니다.
