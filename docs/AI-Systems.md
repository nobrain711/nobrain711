# AI Systems

AI 모델의 설계, 학습, 평가, 배포까지의 전체 생명주기를 정리합니다.  
단순한 모델 구현을 넘어, 재현성과 확장성을 고려한 시스템 관점의 AI 구조를 다룹니다.

이 페이지는 AI Systems의 전체 구조를 설명하는 Overview이며,  
각 세부 영역은 하위 페이지에서 상세히 정리합니다.

---

## [Artificial Intelligence Overview](Artificial-Intelligence-Overview)
AI의 정의와 발전 흐름을 정리하고, 규칙 기반 접근에서 학습 기반 접근으로의 전환을 설명합니다.
- AI의 정의
- Narrow AI vs General AI
- Rule-based vs Learning-based

## [Machine Learning](Machine-Learning)
데이터 기반 학습 모델의 원리와 주요 알고리즘 구조를 다룹니다.
- Supervised / Unsupervised
- Tree Models
- Linear Models
- Ensemble

## [Deep Learning](Deep-Learning)
신경망 기반 모델 구조와 최적화 기법을 정리합니다.
- Neural Networks
- CNN / RNN
- Transformer
- Optimization

## [Model Evaluation](Model-Evaluation)
모델의 성능 측정과 일반화 능력 검증 방법을 다룹니다.
- Metrics (Regression / Classification)
- Cross Validation
- Bias & Variance
- Overfitting

## [MLOps](MLOps)
실험 관리와 재현성 확보를 위한 운영 관점을 정리합니다.
- MLflow
- Reproducibility
- Versioning
- Hyperparameter Logging

## [Model Deployment](Model-Deployment)
학습된 모델을 실제 서비스 환경에 배포하고 운영하는 구조를 다룹니다.
- REST API
- Batch vs Real-time
- Containerization
- Monitoring