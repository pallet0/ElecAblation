---
1. "정확도-해석 가능성 트레이드오프(Accuracy-Interpretability trade-off)는 실재하며, 학계에서 입증되고 수용된 개념입니다."

Rudin (2019) — "Stop Explaining Black Box ML Models for High Stakes Decisions" — Nature Machine Intelligence
트레이드오프 서사에 대한 가장 강력한 비판가조차도, 트레이드오프가 실제로 존재할 때 해석 가능성이 그 비용을 감수할 가치가 있다는 점을 인정합니다. 과학적 질문(어떤 전극이 중요한가?)에 있어서는 해석 가능성 자체가 연구의 기여이며, 단순히 높은 정확도만이 목적은 아닙니다.

Dziugaite, Ben-David & Roy (2020) — "Enforcing Interpretability and its Statistical Impacts" — arXiv:2010.13764
모델에 해석 가능성 제약을 가하는 것이 통계적 위험(excess statistical risk)을 초래할 수 있음을 처음으로 공식화하여 통계적으로 증명했습니다. 트레이드오프는 문제에 따라 다르며 정량화가 가능합니다 — 이는 정확히 본 연구가 처한 상황과 일치합니다.

Lipton (2018) — "The Mythos of Model Interpretability" — Communications of the ACM
해석 가능성은 단일한 개념이 아니라 다각적인 축을 가집니다. 본 모델은 표현력의 한 종류(임의의 채널 간 결합)를 다른 종류의 가치(분해 가능한 채널 중요도)와 교환하는 것입니다.

2. "어텐션(Attention)은 유효한 설명 방법입니다 — 단, 절제 실험(Ablation)을 통해 검증되었을 때에 한합니다."

Jain & Wallace (2019) — "Attention is not Explanation" — NAACL 2019
지도교수님이 반박 자료로 인용할 수도 있는 논문입니다. 하지만 자세히 읽어보면, 검증 없는 어텐션을 무조건 신뢰해서는 안 된다는 의미입니다. 이들은 삭제/절제(Erasure/Ablation) 테스트를 권장하며, 이는 본 연구의 점진적 절제(Progressive ablation) 실험이 수행하는 바와 정확히 일치합니다.

Wiegreffe & Pinter (2019) — "Attention is not not Explanation" — EMNLP 2019
Jain & Wallace에 대한 직접적인 반론입니다. 적대적 어텐션 분포가 간단한 진단을 통과하지 못함을 보여줌으로써, 적절히 테스트된 어텐션은 진정한 설명 가치를 지님을 증명했습니다.

Serrano & Smith (2019) — "Is Attention Interpretable?" — ACL 2019
어텐션의 유효성을 검증하는 올바른 방법으로 삭제 테스트(높은 어텐션 피처를 제거하고 정확도 하락을 측정)를 명시적으로 권장합니다. 본 연구의 점진적 절제가 바로 이 테스트입니다.

3. "절제 실험(Ablation)은 해석 가능성을 검증하는 최적의 표준(Gold Standard)입니다."

DeYoung et al. (2020) — "ERASER: A Benchmark to Evaluate Rationalized NLP Models" — ACL 2020
본 연구가 직접 구현한 두 가지 핵심 지표를 정의합니다:
- 포괄성(Comprehensiveness): 중요한 피처를 제거했을 때 정확도가 하락해야 함 (본 연구의 '상위 어텐션 채널 제거' 곡선)
- 충분성(Sufficiency): 중요한 피처만 남겼을 때 정확도가 유지되어야 함 (본 연구의 '상위 채널만 유지' 곡선)

Li & Janson (2024) — "Optimal Ablation for Interpretability" — NeurIPS 2024
점진적 절제가 중요도 점수(Importance scores)의 유의미성을 테스트하기 위한 이론적으로 원칙 있는 방법론임을 공식화합니다.

4. "EEG 관련 논문들은 이미 이 방식을 채택하고 트레이드오프를 수용하고 있습니다."

Valderrama & Sheoran (2025) — "Identifying Relevant EEG Channels for Subject-Independent Emotion Recognition Using Attention Network Layers" — Frontiers in Psychiatry
SEED 데이터셋에서 어텐션을 사용하여 EEG 채널 중요도를 순위화하고 Wilcoxon 검정으로 검증했습니다. 본 연구와 거의 동일한 방법론을 사용합니다.

Liu et al. (2024) — "ERTNet: An Interpretable Transformer-Based Framework for EEG Emotion Recognition" — Frontiers in Neuroscience
"해석 가능성은 임상 시스템에서 모델을 더 신뢰할 수 있게 만드는 핵심 요소"라고 명시하며, EEG 감정 인식에서 정확도와 해석 가능성 간의 긴장 관계를 인정했습니다.

5. "임상의 및 이해관계자들은 이해를 위해 더 낮은 정확도를 수용합니다."

Tonekaboni et al. (2019) — "What Clinicians Want" — ML4H, PMLR 106
핵심 발견: 임상의들은 "정확도가 다소 낮더라도 성능 저하의 이유를 알 수 있다면 해당 모델을 수용할 것"이라고 답했습니다. 이들은 단순한 수치보다 모델이 언제, 왜 실패하는지 이해하는 것에 더 높은 가치를 둡니다.

---
지도교수님께 제시할 핵심 논거

본 연구의 기여는 단순히 "감정을 잘 분류하는 모델"을 만드는 것이 아닙니다. 그보다는 "4개 클래스 감정 인식을 주도하는 전극이 무엇인지 밝혀내는, 검증된 해석 가능 기제"를 제안하는 데 있습니다. 정확도 격차는 어텐션 제약으로 인해 이론적으로 예상되는, 학계에 문서화된 비용입니다(Dziugaite 2020). 단순한 정확도 수치가 아니라 절제 곡선(Ablation curves)이 연구의 핵심 결과이며, 이는 어텐션의 충실도(Faithfulness)를 검증하는 표준인 ERASER 프레임워크의 포괄성 및 충분성 지표를 구현한 것입니다(DeYoung 2020).

MLP 베이스라인은 변경하지 않고 유지해야 합니다. 이는 해석 가능성 제약 없이 도달할 수 있는 정확도의 상한선(Upper bound)을 보여줌으로써, 어텐션 모델의 성능 차이(~10%)를 정량화하고 정당화할 수 있는 합리적 비용으로 맥락화하는 역할을 합니다.
