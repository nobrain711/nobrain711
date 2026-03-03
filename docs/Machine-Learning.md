# Machine Learning (ML)

## 1. Overview

Machine Learning(ML)은 **데이터로부터 패턴을 학습하여 예측 또는 의사결정을 수행하는 알고리즘적 방법론**이다.

전통적인 프로그래밍이 사람이 규칙을 명시적으로 작성하는 방식이라면  
ML은 **데이터를 통해 규칙을 자동으로 학습한다는 점**이 핵심 차이이다.

### 기본 수식

$$
f(X) \rightarrow y
$$

- $X$: 입력 데이터 (Feature)
- $y$: 출력 값 (Label)
- $f$: 학습된 함수 (Model)

---

## 2. Learning Paradigms (학습 방법)

ML은 **데이터(Feature)와 정답(Label)의 존재 여부에 따라 구분**된다.

---

### 2.1 Supervised Learning (지도 학습)

입력 데이터 $X$와 정답 $y$가 함께 주어진 상태에서 학습하는 방식이다.  
입력과 출력 사이의 관계를 학습하여 새로운 데이터에 대해 예측한다.

---

#### 2.1.1 Classification (분류)

입력 데이터에 대해 **이산적인 범주(Label)** 를 예측하는 문제이다.

##### 수식

$$
y \in \{c_1, c_2, \dots, c_k\}
$$

- $y$: 모델이 예측해야 하는 출력 값
- $\{c_1, c_2, \dots, c_k\}$: 예측 가능한 클래스 집합
- $k$: 전체 클래스 개수

---

##### 2.1.1.1 Binary Classification (이진 분류)

$$
y \in \{0,1\}
$$

예시
- 스팸 ($1$) / 정상 ($0$)
- 질병 있음 ($1$) / 없음 ($0$)

---

##### 2.1.1.2 Multi-class Classification (다중 분류)

$$
y \in \{cat, dog, bird\}
$$

예시
- 고양이
- 개
- 새

---

#### 2.1.2 Regression (회귀)

입력 데이터에 대해 **연속적인 실수 값(Continuous Value)** 을 예측하는 문제이다.  
출력 값의 크기와 차이가 의미를 가진다.

##### 수식

$$
y \in \mathbb{R}
$$

- $y$: 모델이 예측해야 하는 연속적인 출력 값
- $\mathbb{R}$: 실수 집합 (Real Number Set)

---

##### 특징

- 출력은 연속적인 수치 값
- 오차(Error)를 최소화하는 방향으로 학습
- 값의 차이가 의미를 가짐

---

##### 대표 예시

- 주택 가격 예측
- 주가 예측
- 온도 예측
- 매출 예측

---

##### Classification vs Regression 비교

|   구분   |       Classification       |     Regression      |
|:------:|:--------------------------:|:-------------------:|
| 출력 형태  |     범주형 (Categorical)      |  연속형 (Continuous)   |
|   수식   | $y \in \{c_1, ..., c_k\}$  | $y \in \mathbb{R}$  |
|   예시   |           스팸/정상            |        가격 예측        |

---

#### 2.1.3 Algorithms Used in Supervised Learning

Supervised Learning에서는   
**문제 유형(Classification / Regression)에 따라서 다양한 알고리즘**이 사용된다.

---

##### 2.1.3.1 K-Nearest Neighbors (KNN)
![KNN Illustration](https://mlarchive.com/wp-content/uploads/2022/09/img2-3-1024x585.png)
가장 직관적인 거리 기반 알고리즘이다.  
**주변 데이터와의 거리를 계산하여 예측**을 수행한다.

- Classification과 Regression 모두 가능
- **거리 척도(Euclidean 등)에 의존**
- 비모수(Non-parametric) 모델
- **훈련 데이터를 메모리에 저장해두고 예측 시 직접 참조 (Instance-based Learning)**
---
###### 기본 수식
$$
d(x, x_i) = \sqrt{\sum_{j=1}^{n} (x_j - x_{ij})^2}
$$
* $x$: 예측하려는 데이터
* $x_i$: 훈련 데이터
* $n$: $feature$(특성) 개수
* $d(x, x_i)$: 두 데이터 사이의 거리

###### 회귀 수식 (KNN Regression)
회귀의 경우에는 **가장 가까운 $k$개의 이웃을 평균으로 예측**합니다
$$
\hat{y} = \frac{1}{k} \sum_{i \in N_k(x)} y_i
$$

* $\hat{y}$: 예측값
* $N_k(x)$: $x$와 가장 가까운 $k$개의 이웃 집합
* $y_i$: 각 이웃의 실제 값

---

##### 2.1.3.2 Linear Models
입력과 출력 사이의 선형 관계를 가정하는 모델이다.

---

###### Linear Regression
![Linear model](https://contenthub-static.grammarly.com/blog/wp-content/uploads/2024/09/156443-61046105blogvisuals-Linear-regression1.png)

- 기본 선형 회귀 모델
- 최소제곱법 기반

###### 모델 수식

$$
\hat{y} = Xw + b
$$

- $\hat{y}$ : 예측값
- $X$ : 입력 데이터($feature$) $vector\ /metices$
- $w$ : 가중치(각 $feature$의 영향력)
- $b$ : 절편(입력이 $0$일 때의 기본 출력값)

###### 가중치(Weight)와 절편(Bias)

- $w$: 각 $feature$가 출력에 미치는 영향력 (기울기 역할)
- $b$: 입력이 $0$일 때의 기본 출력값 (위치 조정 역할)

가중치는 모델의 기울기를 결정하고
절편은 모델의 위치를 결정한다.

---

###### 특징

- 해석이 용이
- 계산 비용이 낮음
- 선형 관계를 가정

---

##### 2.1.3.3 Logistic Regression
![Logistic](https://itfeature.com/wp-content/uploads/2015/01/Logistic-Regression.jpg)
선형 모델 기반의 이진 분류(Binary Classification) 알고리즘이다.  
선형 결합 결과를 확률로 변환하여 분류를 수행한다.

---

###### 모델 수식

$$
z = Xw + b
$$

- $z$ : 선형 결합 값(점수, score)
- $X$ : 입력 데이터($feature$) $vector\ /행렬$
- $w$ : 가중치 $vector$ 
- $b$ : 절편$(Bias)$

입력 데이터를 하나의 점수(Score)로 변환하는 과정이다.   
전개하면 다음과 같다.

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$
- $x_j$ : $j$번째 $feature$ 값
- $w_j$ : $j$번째 $feature$의 가중치
- $n$ : $feature$ 개수
---

###### 확률 변환 (Sigmoid)

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

- $\sigma(z)$ : $sigmoid $함수
- $\hat{y}$ : 클래스가 $1$일 확률($0$과 $1$ 사이 값) 
선형 결합 값을 $0$과 $1$ 사이의 확률 값으로 변환한다.

---

###### 결정 기준 (Decision Rule)

$$
\hat{y} =
\begin{cases}
1 & \text{if } \sigma(z) \ge 0.5 \\
0 & \text{otherwise}    
\end{cases}
$$

- 예측 확률이 $0.5$ 이상이면 $1$, 미만이면 $0$으로 분류한다.

---

###### 특징

- 선형 Decision Boundary
- 확률 해석 가능
- 선형 모델 기반 분류기