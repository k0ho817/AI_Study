# AI_Study
파이썬으로 만드는 인공지능 요약

# Chapter01. 인간 지능을 흉내 내는 인공지능

## 1.2 인공지능을 바라보는 관점
### 1.2.1 인공지능에 대한 정의
> **지능** : 계산이나 문장 작성 따위의 지적 작업에서, 성취 정도에 따라 정하여지는 적응 능력

지적 작업에는 학습, 인식, 추론, 창작 등이 포함된다.   
학습은 경험이 쌓임에 따라 점점 성능이 좋아지는 지적 작업이고,  
인식은 오감을 통해 외부 환경의 상태를 알아내는 지적 작업이며,  
추론은 이미 알고 있는 사실을 바탕으로 새로운 사실 또는 새로운 지식을 알아내는 지적 작업이다.  
창작이란 세상에 없던 새로운 것을 만들어내는 지적 작업이다.  
또한 '적응'이란 인식이나 추론을 통해 알아낸 사실을 바탕으로 변화하는 환경에 자신을 맞추는 능력을 말한다. 

> **표준국어대사전의 인공지능에 대한 정의**  
**인공지능** : 인간의 지능이 가지는 학습, 추리, 적응, 논증 따위의 기능을 갖춘 컴퓨터 시스템

> **풀(Poole)이 정의한 인공지능**  
**Artificial intelligence** : The field that studies the synthesis and analysis of computational agents that act initelligently  
***인공지능*** : *지능적으로 행동하는 계산 에이전트를 만들고 분석하는 학문 분야*

#### 풀의 에이전트 지능 행위
에이전드란 주어진 환경에서 주어진 목표를 향해 행동하는 것이다.
- 환경과 자신의 목표에 맞게 적절한 행동을 수행
- 변화하는 환경과 목표에 유연하게 대처
- 과거 경험으로부터 학습
- 인식과 계산 능력의 한계에 적절하게 대처

# Chapter03. 기계학습과 인식
## 기계 학습 기초
### 3.1.1 데이터셋 읽기
#### iris 데이터셋 읽기
```python
from sklearn import datasets    # scikit-learn dataset module 불러오기

d = datasets.load_iris()        # iris dataset 변수에 저장
print(d.DESCR)                  # iris dataset 정보 출력
```
출력
```
.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
                
    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.

|details-start|
**References**
|details-split|

- Fisher, R.A. "The use of multiple measurements in taxonomic problems"
  Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
  Mathematical Statistics" (John Wiley, NY, 1950).
- Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
  (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
- Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
  Structure and Classification Rule for Recognition in Partially Exposed
  Environments".  IEEE Transactions on Pattern Analysis and Machine
  Intelligence, Vol. PAMI-2, No. 1, 67-71.
- Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
  on Information Theory, May 1972, 431-433.
- See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
  conceptual clustering system finds 3 classes in the data.
- Many, many more ...

|details-end|
```
위의 출력 결과를 보면 iris 데이터는 Setosam Versicolor, Virginina 세 부류로 구성되는 것을 알 수 있다.  
부류별로 50개씩 총 150개의 sample이 있고, 각 sample은 sepal length, sepal widath, petal length, petal width 4가지의 변수로 표현된다.  
기계 학습에서는 각 변수를 ***특징(feature)*** 이라 부른다. 샘플 하나에 여러 개의 특징이 있으므로 이는 벡터로 표현되는데 이 벡터를 ***특징 벡터(feature vector)*** 라고 부른다.

```python
for i in range(len(d.data)):            # d.data의 길이만큼 반복 실행 i = [0,1,2,...,149]
    print(i+1, d.data[i], d.target[i])  # i+1 값, 특징벡터, 레이블 출력
```
출력
```
1 [5.1 3.5 1.4 0.2] 0
2 [4.9 3.  1.4 0.2] 0
3 [4.7 3.2 1.3 0.2] 0
...
51 [7.  3.2 4.7 1.4] 1
52 [6.4 3.2 4.5 1.5] 1
53 [6.9 3.1 4.9 1.5] 1
...
101 [6.3 3.3 6.  2.5] 2
102 [5.8 2.7 5.1 1.9] 2
103 [7.1 3.  5.9 2.1] 2
...
148 [6.5 3.  5.2 2. ] 2
149 [6.2 3.4 5.4 2.3] 2
150 [5.9 3.  5.1 1.8] 2
```
위의 코드는 데이터의 각각의 sample을 출력한다.  
i+1은 sample의 index를 뜻하고, d.data[i]는 i번째 sample의 특징 벡터를 보여준다. 마지막으로 d.target[i]는 i번째 데이터가 어디에 속하는지 알려준다.  
d.target[i]처럼 데이터가 어디에 속하는지 알려주는 것을 ***레이블(label)*** 또는 ***참값(ground truth)*** 이라고 한다.

#### 기계 학습에서 데이터셋의 표현
하나의 샘플은 d개의 특징을 갖는 특징 벡터로 표현된다. ***d차원의 특징 벡터는 $\mathbf{x} = \mathrm{(x_1, x_2, x_3, \dots ,x_d)}$ 으로 표기한다.***  
***샘플의 레이블은 $\mathrm{y}$로 표기한다.*** 부류의 개수를 $\mathrm{c}$라 하면, $\mathrm{y}$는 0, 1, 2, ..., c-1중의 한 값 또는 1, 2, 3, ..., c 중의 한 값을 갖는다.  
기계학습은 때때로 ***원 핫 코드 (one-hot code)*** 로 $\mathrm{y}$를 표기한다. 원 핫 코드는 요소 하나만 1인 이진 벡터이다. 예를 들어 iris 데이터셋에서 Setosa에 속하는 샘플은 (1,0,0), Versicolor 샘플은 (0,1,0), Virginica 샘플은 (0,0,1)로 표현한다.  
때때로 다중 레이블을 허용하는 경우도 있다. 예를 들어 자연 영상에 고양이와 쥐가 같이 들어있는데, 고양이는 두 번째 부류이고 쥐는 다섯 번째 부류라면 레이블은 (0,1,0,0,1,0,...,0)이다.
> **데이터 프레임**  
행은 샘플을 나타내고 열은 속성을 표현하는 자료구조 이다.. pandas 라이브러리는 이 자료구조를 데이터 프레임(data fram)이라 하며, 데이터 프레임에서 수행할 수 있는 함수를 풍부하게 제공한다.

### 3.1.2 기계 학습 적용 : 모델링과 예측

SVM을 이용한 모델링과 예측
``` python
from sklearn import svm                                 # scikit-learn 모듈에서 svm 불러오기

s=svm.SVC(gamma=0.1, C=10)                              # svm 분류 모델 SVC 객체 생성
s.fit(d,data, d.target)                                 # iris 데이터로 학습

new_d = [[6.4, 3.2, 6.0, 2.5], [7.1, 3.1, 4.7, 1.35]]   # 새로운 샘플데이터 생성

res = s.predict(new_d)                                  # SVC객체를 이용하여
                                                        # 새로운 데이터의 레이블 예측
print(f"새로운 2개 샘플의 레이블은 {res}")              # 출력
```
출력
```
새로운 2개 샘플의 부류는  [2 1]
```
10행에서 fit 함수를 사용해 모델을 학습한다. fit 함수는 데이터를 가지고 학습을 하는데,  
이때 사용하는 데이터를 ***훈련 집합(train set)*** 이라고 한다. 훈련 집합은 샘플의 특징 벡터와 샘플의 레이블을 제공한다.

학습을 마치면 학습된 모델을 이용하여 ***예측(prediction)*** 을 할수 있다.   
예측이란 샘플의 부류를 알아내는 작업이므로 샘플을 인식한다고 말할 수 있다.
> **하이퍼 매개변수 설정**  
하이퍼 매개변수(hyper parameter)란 모델의 동작을 제어하는 데 쓰는 변수이다. 모델의 학습을 시작하기 전에 설정해야 하는데, 적절한 값으로 설정해야 좋은 성능을 얻을 수 있다. 최적의 하이퍼 매개변수 값을 자동으로 설정하는 일을 하이퍼 매개변수 최적화(hyper parameter optimization)라 하는데, 이것은 기계 학습의 중요한 주제 중 하나다.

## 3.2 인공지능 제품의 설계와 구현

### 3.2.1 인공지능 설계

#### 데이터 확보
한가지 특성에 지나치게 치우치게 데이터를 수집해서 정확률에 부정적인 요인을 만드는 경우를 ***데이터의 편향(data bias)*** 이 있다고 말한다. 데이터 수집 단계에서는 가급적 데이터 편향이 적게 발생하도록 주의 해야한다.

#### 데이터의 저장
데이터를 저장할때는 원하는 레이블로 잘 구별하도록 ***분별력(discriminative power)*** 높은 특징을 선택해야 한다.

### 3.2.2 규칙 기반 vs 고전적 기계 학습 vs 딥러닝

기계 학습의 데이터를 만드는 과정, 즉 특징 벡터를 추출하고 레이블을 붙이는 과정은 규칙 기반과 같다. 하지만 구칙을 만드는 일은 기계 학습 모델을 이용해 자동으로 수행한다. 기계학습에서는 규칙을 만드는 일을 ***모델링(modeling)*** 이라고 한다.  
딥러닝의 경우 데이터를 준비하는 과정이 훨씬 쉽다. 레이블은 고전적 기계 학습과 마찬가지로 전문가의 손을 거쳐 만든다. 하지만 특징 벡터는 학습이 자동으로 알아낸다.

> **원샷 학습, 퓨샷 학습, 준지도 학습**  
레이블을 붙이는 작업은 전문가가 해야 하므로 비용이 많이 든다. 따라서 딥러닝에서는 레이블이 있는 샘플을 하나만 사용해 학습하는 원샷 학습(1-shot learning), 몇 개의 샘플만 사용하는 퓨샷 학습 (few-show learning), 레이블이 있는 소량의 샘플과 레이블이 없는 대량의 샘플을 같이 사용하는 준지도 학습(semi-supervised learning)을 활용한다.

규칙 기반과 고전적 기계 학습에서는 특징을 사람이 설계하거나 추출한다. 이런 종류의 특징을 수작업 특징(hand-crafted feature)이라고 한다.  
딥러닝에서는 특징을 자동으로 학습하는데, 이 과정을 특징 학습(feature learning) 또는 표현 학습(representation learning)이라고 한다.

## 3.3 데이터에 대한 이해
### 3.3.1 특징 공간에서 데이터 분포
iris는 특징이 4개이므로 4차원 특징 벡터로 표현된다. 이는 4차원 특징 공간(feature space)을 구성한다고 말하기도 한다. 다음은 iris dataset을 1개의 특징을 제외하고 3차원 공간에 나타내는 코드이다.

```python
import plotly.express as px

df = px.data.iris()
# petal_length를 제외한 3차원 공간 구성
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')
fig.show(rederer='browser')
```
출력
![3dim image](https://github.com/k0ho817/AI_Study/assets/53679360/5859fc1b-12cb-4040-bff7-1e7470e553e6)

> **다차원 특징 공간**  
종이에 그릴 수 있는 공간은 3차원으로 제한되지만, 수학은 아주 높은 차원까지 다룰 수 있다. 예를 들어 2차원 상의 두 점 $\mathbf{x}=\mathrm{(x_1, x_2)}$와 $\mathbf{y}=\mathrm{(y_1, y_2)}$의 거리를 $d\mathbf{(x,y)}=\mathrm{\sqrt{(x_1-y_1)^2+(x_2-y_2)^2}}$으로 계산할 수 있는데, 4차원 상의 두 점 $\mathbf{x}=\mathrm{(x_1,x_2,x_3,x_4)}$와 $\mathbf{y}=\mathrm{(y_1,y_2,y_3,y_4)}$의 거리는 $d\mathbf{(x,y)}=\mathrm{\sqrt{(x_1-y_1)^2+(x_2-y_2)^2+(x_3-y_3)^2+(x_4-y_4)^2}}$로 계산할수 있다.  
일반적으로 $d$차원 상의 두 점의 거리는 $d\mathbf{(x,y)}= \sqrt{\displaystyle\sum_{i=1}^{d} (\mathrm{x}_i-\mathrm{y}_i)^2}$로 계산한다. 기계 학습에서는 $d$=수백 ~ 수만에 달하는 매우 고차원 특징 공간의 데이터를 주로 다룬다.

### 3.3.2 연상 데이터 사례 : 필기 숫자
MNIST 데이터셋은 미국 국립표준기술연구소(NIST)에서 미국인을 대상으로 수집한 필기 숫자 데이터이다.
MNIST에는 7만 자가 포함되어 있는데, 숫자는 0,1,2, ..., 9의 10개 부류이므로 부류당 7천개의 샘플이 있는 셈이다. 샘플은 28 X 28 맵으로 표현되어 있다. 샘플 하나가 총 784개의 화소로 구성되어있음을 알 수 있다. 한 화소는 [0,255]사이의 명암값으로 표현된다.

```python
from sklearn import datasets
import matplotlib.pyplot as plt

digit = datasets.load_digits()  #데이터들을 digit변수에 저장

plt.figure(figsize=(5,5))       #화면 사이즈 설정
# 첫번째에 해당하는데이터
plt.imshow(digit.images[0], cmap=plt.cm.gray_r, interpolation="nearest")

plt.show()
print(digit.data[0])
print(f"이 숫자는 {digit.target[0]} 입니다.")
```
출력  
<img src="https://github.com/k0ho817/AI_Study/assets/53679360/b2b5fbd3-33f4-4f45-823e-f68a71b6c4fb" width="300">
```
[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
 15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
  0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
  0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
이 숫자는  0 입니다.
```

> **matplotlib을 이용한 시각화**  
파이썬에서 matplotlib 라이브러리는 시각화에 가장 널리 쓰인다. 인공지능은 학습 과정이나 예측 결과를 시각화하는 데 matplotlib을 자주 사용한다. [matplotlib의 공식 사이트에서 제공하는 튜토리얼 문서](https://matplotlib.org/stable/tutorials/index)를 숙지하고 넘어간다.

### 3.3.3 영상 데이터 사례 : lfw 얼굴 데이터셋
lfw(labeled faces in the wild)는 유명인의 얼굴 영상을 모아둔 데이터셋이다. 5,749명에 대한 13,233장의 영상이 들어 있다. 영상 한 장은 50 X 37 맵으로 표현되며 한 화소는 [0,255]사이의 명암값으로 표현된다. 이 데이터셋에는 사람당 평균 두 장 남짓한 영상이 있기 때문에 얼굴 인식보다는 얼굴 영상 두 장이 주어졌을 때 같은 사람인지 알아내는 얼굴 검증에 주로 사용된다.

```python
from sklearn import datasets    # 데이터셋 불러오기
import matplotlib.pyplot as plt # matplot으로 데이터 시각화

# 데이터 불러오기
lfw = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)

plt.figure(figsize=(20,5))      # matplot 캔버스 크기 지정

for i in range(8):
    plt.subplot(1,8,i+1)
    plt.imshow(lfw.images[i],cmap=plt.cm.bone)
    plt.title(lfw.target_names[lfw.target[i]])

plt.show()
```
출력
![lfw](https://github.com/k0ho817/AI_Study/assets/53679360/605d4c65-3336-4ae1-bedc-1cfc3a2fc73e)
lfw 데이터 편향이 존재한다. 어린이 사진이 매우 적고, 갓난아기는 아예 없으며, 80세 이상이 매우 적고, 여성이 상대적으로 적으며, 아예 없거나 매우 희소한 인종이 많기 때문이다. 이런 편향은 유명인 위주로 사진을 모았기 때문에 발생한 것이다.

### 3.3.4 텍스트 데이터 사례 : 20newsgroups
20newsgroups 데이터셋은 웹에서 수집한 문서를 2-개 부류로 구분한 텍스트를 담고 있다.
```python
from sklearn import datasets
import matplotlib.pyplot as plt

news = datasets.fetch_20newsgroups(subset='train') # dataset read   
print(f"******\n{news.data[0]}\n******")
print(f"이 문서의 부류는 <{news.target_names[news.target[0]]}> 입니다.")
```
출력
```
******
From: lerxst@wam.umd.edu (where's my thing)
Subject: WHAT car is this!?
Nntp-Posting-Host: rac3.wam.umd.edu
Organization: University of Maryland, College Park
Lines: 15

 I was wondering if anyone out there could enlighten me on this car I saw
the other day. It was a 2-door sports car, looked to be from the late 60s/
early 70s. It was called a Bricklin. The doors were really small. In addition,
the front bumper was separate from the rest of the body. This is 
all I know. If anyone can tellme a model name, engine specs, years
of production, where this car is made, history, or whatever info you
have on this funky looking car, please e-mail.

Thanks,
- IL
   ---- brought to you by your neighborhood Lerxst ----
******
이 문서의 부류는 <rec.autos> 입니다.
```
20newsgroups 데이터셋은 부류 정보를 가지고 있으므로 앞서 살펴본 데이터와 마찬가지로 분류 문제에 해당한다. 문서가 다른 샘플과 다른 점은 가변길이라는것이다. 또한 문서는 단어가 나타나는 순서가 매우 중요한 시계열 데이터(time-series data)에 속한다.
## 3.4 특징 추출과 표현
기계학습의 과정은 전형적으로 다음과 같이 이뤄진다.
> 데이터 수집 $\to$ 특징 추출 $\to$ 모델링 $\to$ 예측  
딥러닝은 특징 추출과 모델링을 동시에 최적화해준다, 고전적인 기계 학습을 사용한다면 특징 추출을 사람이 설계하고 구현해야 한다. 어떤 경우든 특징은 분별력이 높아야 한다.

### 3.4.1 특징의 분별력

사람은 직관적으로 ***분별력(discriminative power)*** 이 높은 특징을 선택해 사용한다.  
기계학습도 사람처럼 분별력이 높은 특징을 추출해야 한다.  
딥러닝은 영상에서 자동으로 최적의 특징을 추출해준다.  
![특징의 분별력](https://github.com/k0ho817/AI_Study/assets/53679360/bf02cb1a-d33d-4652-8eb7-0d688bdf1fa4)  
a는 ***선형 모델(linear model)*** 로 쉽게 분류할 수 있는 상황이고, b는 ***비선형 모델(nonlinear model)*** 로 분류할수 있다. c, d는 일정한 양의 오류를 허용해야 하는 복잡한 상황이다. 우리가 사는 세상은 c,d와 같이 오류를 허용할 수밖에 없는 데이터를 생성한다. ㄷ데이터의 원천적인 성질이 그럴 수도 있고, 데이터 수집 과정에서 사람이 측정이나 레이블링을 잘못해서, 또는 특징 설계를 잘못해서 그렇게 분포되었을 수도 있다.
기계학습은 c,d와 같은 데이터를 처리해야 한다. 기계 학습에서 특징 추출 알고리즘이 해야 하는 일은 가급적 d와 같은 상황은 피하고 c와 같은 상황을 만들어내는 것이다. 딥러닝이 고전적인 기계 학습보다 뛰어난 점을 감안하면 딥러닝은 d보다 c에 가까운 특징 공간을 형성한다고 볼 수 있다.

### 3.4.2 특징 값의 종류
특징들이 거리 개념이 있는경우 ***수치형 특징(numerical feature)*** 이라고 한다.
특징들이 어떠한 범주로 나눠지는 경우 ***범주형 특징(categorical feature)*** 이라고 한다.

범주형 특징은 크게 ***순서형(ordinal)*** 과 ***이름형(nominal)*** 으로 나뉘는데, 순서에 의미가 있는 범주는 순서형이고, 순서에 의미가 없는 범주는 이름형이다.

순서형은 거리 개념이 있고 순서대로 정수를 부여하면 수치형으로 취급할 수 있다.

이름형은 거리 개념이 없고 보통 원핫 코드(one-hot code)로 표현한다.

## 3.5 필기 숫자 인식

특징 추출을 위한 코드를 작성하고, sklearn 라이브러리가 제공하는 fit함수로 모델링하고, predict 함수로 예측하는 연습

### 화솟값을 특징으로 사용
필기 숫자를 인식하기 위해서는 화소 각각을 특징으로 간주한다. 샘플 하나는 8 X 8 맵으로 표현되므로 64개의 특징이 있고, 결국 64차원 특징 벡터가 생긴다. 2차원 구조 형태로 표현된 패턴을 벡터 형태로 변환한다. 이때 행 우선 이어 붙이기 방식을 사용한다. 0행 뒤에 1행, 1행뒤에 2행을 이어 붙이는 방식이다.

> **컴퓨터 프로그래밍에 쓰이는 패턴의 중요성**  
컴퓨터 프로그래밍에도 패턴이 있다. 이 패턴을 잘 기억하고 따라 하는 것은 좋은 프로그래머로 성장하는 지름길이다.

```python
from sklearn import datasets
from sklearn import svm

digit = datasets.load_digits()

# svm의 분류기 모델 SC를 학습
s = svm.SVC(gamma=0.1, C=10)
s.fit(digit.data, digit.target)

# 훈련 집합의 앞에 있는 샘플 3개를 새로운 샘플로 간주하고 인식해봄
new_d = [digit.data[0], digit.data[1],digit.data[2]]
res = s.predict(new_d)
print(f'예측값은 {res}')
print(f'참값은 {digit.target[0], digit.target[1],digit.target[2]}')

# 훈련 집합을 테스트 집합으로 간주하여 인식해보고 정확률을 측정
res = s.predict(digit.data)
# 아래 코드와 같음
correct = [i for i in range(len(res)) if res[i]==digit.target[i]]
# correct = []
# for i in range(len(res)):
#     if res[i] == digit.target[i]:
#         correct.append(i)
accuracy = len(correct)/len(res)    # 일치하는 값을 일치값으로 나눔
print(f"화소 특징을 사용했을 때 정확률 = {accuracy*100}%")
```
출력
```
예측값은 [0 1 2]
참값은 (0, 1, 2)
화소 특징을 사용했을 때 정확률 = 100.0%
```
정확률이 100%인 이유는 데이터가 단순하고 학습에 활용한 훈련 집합으로 테스트했기 때문이다.

## 3.6 성능 측정
기계 학습에는 SVM뿐만 아니라 K-NN, 결정트리(decision tree), 랜덤 포리스트(random forest), 신경망, 딥러닝 등 아주 많은 모델을 사용할 수 있다, 이 작업을 ***모델 선택(model selection)*** 이라고 한다. 모델 선택을 하려면 여러 모델의 성능을 견주어 가장 좋은 것을 선택해야 하기 때문에 객관적인 성능 측정이 매우 중요하다.

완성된 시스템을 현장에 설치할지 여부를 판단할 때에도 성능 측정은 핵심적인 사항이다. 성능을 제대로 평가하려면 적절한 평가 기준이 있어야 한다. 또한 모델을 학습하는 과정에서 사용한 훈련 집합 외의 새로운 데이터 집합으로 성능을 측정해야 한다. 이처럼 새로운 데이터로 성능을 측정하는 일을 ***일반화(generalization) 능력 측정*** 이라고 한다.

### 3.6.1 혼동돈행렬과 성능 측정 기준
혼동행렬(confusion matrix)은 부류별로 옳은 분류와 틀린 분류의 개수를 기록한 행렬이다. 긍정을 긍정으로 예측하면 ***참 긍정(TP_True Positive)*** , 긍정을 부정으로 잘못 예측하면 ***거짓 부정(FN_False_negative)*** , 부정을 긍정으로 잘못 예측하면 ***거짓 긍정(FP_False_Positive)*** , 부정을 부정으로 예측하면 ***참 부정(TN_True Negative)***이라 부른다.
<table width="40%" style="display: inline-block;">
    <tr>
        <td rowspan="2" colspan="2"></td>
        <th colspan="6" style="text-align: center; ">참값(그라운드 트루스)</th>
    </tr>
    <tr>
        <td>부류1</td>
        <td>부류2</td>
        <td>$\dots$</td>
        <td>부류j</td>
        <td>$\dots$</td>
        <td>부류c</td>
    </tr>
    <tr>
        <th rowspan="6" style="writing-mode: vertical-lr;">예측한 부류</th>
        <td>부류1</td>
        <td>$n_{11}$</td>
        <td>$n_{12}$</td>
        <td></td>
        <td>$n_{1j}$</td>
        <td></td>
        <td>$n_{1c}$</td>
    </tr>
    <tr>
        <td>부류2</td>
        <td>$n_{21}$</td>
        <td>$n_{22}$</td>
        <td></td>
        <td>$n_{2j}$</td>
        <td></td>
        <td>$n_{2c}$</td>
    </tr>
    <tr>
        <td>$\dots$</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>부류i</td>
        <td>$n_{i1}$</td>
        <td>$n_{i2}$</td>
        <td></td>
        <td>$n_{ij}$</td>
        <td></td>
        <td>$n_{ic}$</td>
    </tr>
    <tr>
        <td>$\dots$</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>부류c</td>
        <td>$n_{c1}$</td>
        <td>$n_{c2}$</td>
        <td></td>
        <td>$n_{cj}$</td>
        <td></td>
        <td>$n_{cc}$</td>
    </tr>
</table>

<table width="40%" style="display: inline-block;">
    <tr>
        <td rowspan="2" colspan="2"></td>
        <th colspan="6" style="text-align: center; ">그라운드 트루스</th>
    </tr>
    <tr>
        <td>긍정</td>
        <td>부정</td>
    </tr>
    <tr>
        <th rowspan="6" style="writing-mode: vertical-lr;">예측값</th>
        <td>긍정</td>
        <td>TP</td>
        <td>FP</td>
    </tr>
    <tr>
        <td>부정</td>
        <td>FN</td>
        <td>TN</td>
</table>  

가장 널리 쓰이는 성능 측정 기준은 ***정확률(accuracy)*** 이다.
$$정확률 = \frac{맞힌 샘플 수}{전체 샘플 수}=\frac{대각선 샘플 수}{전체 샘플 수}$$

정확률은 가장 널리 쓰이지만, 2부류 분류 문제에서는 종종 한계를 드러낸다. 예를 들어 의사가 환자를 진료하는 경우, 정산인이 암환자보다 훨씬 많기 때문에 모두 정상인이라고 진단해도 정확률이 꽤 높다. 가령 암 환자가 200명에 한 명 꼴인 상황에서 모두 정상으로 판정하면 정확률이 99.5%가 된다. 따라서 의사의 환자 진료와 같은 2부류 분류 문제에서는 ***특이도(specificity)*** 와 ***민감도(sensitivity)*** 를 성능 기준으로 사용한다.
$$특이도 = \frac{TN}{TN+FP}, 민감도=\frac{TP}{TP+FN}$$
> 특이도: 참값이 부정인것 중에 예측,참 모두 부정, 민감도: 참값이 긍정인것 중에 예측,참 모두 긍정

웹에서 정보 검색을 수행하거나 영상에서 물체 검출을 하는 경우에는 ***정밀도(precision)*** 와 ***재현율(recall)*** 을 주로 사용한다.
$$정밀도 = \frac{TP}{TP+FP}, 민감도=\frac{TP}{TP+FN}$$
> 정밀도: 예측값이 긍정인것 중에 예측, 참 모두 긍정, 민감도: 참값이 긍정인것 중에 예측, 참 모두 긍정

### 3.6.2 훈련/검증/테스트 집합으로 쪼개기

객관적인 성능 측정을 위한 한 가지 좋은 방법은 데이터를 적절한 비율로 ***훈련 집합(train set)*** , ***검증 집합(validation set)*** , ***테스트 집합(test set)*** 으로 나누어 사용하는 것이다. 여러 모델을 학습하고 성능을 비교할 때는 훈련 집합과 검증 집합을 사용한다. 모델별로 훈련 후 검증 집합을 사용할때 정확률을 측정하고 정확률이 가장 높은 모델을 최종 선택하여 모델 선택을 마친다. 이렇게 선택된 모델에 대해 이제껏 한 번도 사용하지 않고 남겨둔 테스트 집합으로 성능을 측정한다. 이 성능이 일정 기준을 넘으면 합격 판정을 하고 현장에 설치한다.

```python
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
digit = datasets.load_digits()
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size=0.6)

# svm의 분류 모델 SVC를 학습
s = svm.SVC(gamma=0.001)
s.fit(x_train, y_train)

res = s.predict(x_test)

# 혼동 행렬 구함
conf = np.zeros((10,10))
for i in range(len(res)):
    conf[res[i]][y_test[i]]+=1
print(conf)

# 정확률 측정하고 출력
no_correct = 0
for i in range(10):
    no_correct += conf[i][i]
accuracy = no_correct/len(res)
print(f'테스트 집합에 대한 정확률은 {accuracy*100}%입니다.')
```
출력
```
[[54.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0. 74.  0.  0.  0.  0.  1.  0.  0.  0.]
 [ 0.  0. 63.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0. 81.  0.  0.  0.  0.  0.  1.]
 [ 0.  0.  0.  0. 75.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  1.  0. 70.  0.  0.  0.  2.]
 [ 0.  0.  0.  0.  0.  0. 77.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0. 71.  0.  1.]
 [ 0.  0.  0.  0.  1.  0.  0.  0. 66.  0.]
 [ 0.  0.  0.  0.  0.  1.  0.  0.  0. 80.]]
테스트 집합에 대한 정확률은 98.88734353268428%입니다.
```
데이터를 훈련 집합과 테스트 집합의 두 부분으로 나눈다음 , 훈련 집합으로 SVM 모델을 학습하고 테스트 집합으로 성능을 측정하는 프로그래밍을 한다. 전체 데이터를 훈련 집합과 테스트 집합으로 분할하고 테스트 집합으로 혼동 행렬과 정확률을 측정한다.

> **난수를 사용하기 때문에 실행할 때마다 다른 결과가 나오는 프로그램**  
위의 프로그램은 실행할 때마다 출력이 다르게 나온다. train_test_split 함수가 난수를 사용해 데이터를 분할하기 때문이다. 앞으로 등장하는 프로그램에서도 난수를 사용하는 경우가 있는데 마찬가지로 실행할 때마다 다른 결과를 얻게 된다. 동일한 실행 결과를 얻으려면 8행 이전에 np.random.seed(0)을 추가하면 된다. 매개변수 -은 다른 값을 사용해도 된다. 어떤 값이든 고정시키면 매번 같은 난수 열이 생성된다.

### 3.6.3 교차 검증
샘플링을 랜덤하게 하게 된다면 우연히 높은 정확률을 얻게 분할될 수도 있고 낮은 정확률을 얻게 분할될 수도 있다. 이런 우연성을 줄이는 방법은 여러 번 분할해 각각의 성능을 측정하고 평균을 구하는 것이다.

***k-겹 교차검증(k-fold cross validation)*** 은 훈련 집합을 같은 크기의 부분집합 k개로 나눈다. 첫 번째 시도에서는 첫 번째 부분집합을 남겨두고 나머지 4개로 모델을 학습한다. 그리고 남겨둔 첫 번째 부분 집합을 테스트 집합으로 간주하고 성능을 측정한다. 이어서 두 번째, 세 번째, 네 번째, 다섯 번째 부분집합을 테스트 집합으로 간주하고 이 과정을 반복한다. 이렇게 얻은 정확률 5개의 평균을 최종 정확률로 취한다.

```python
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

digit = datasets.load_digits()
s = svm.SVC(gamma=0.001)
accuracies = cross_val_score(s, digit.data, digit.target, cv=5) # 5-겹 교차 검증

print(accuracies)
print(f'정확률(평균) = {accuracies.mean()*100:0.3f}, 표준편차 = {accuracies.std():0.3f}')
```

출력
```
[0.975      0.95       0.98328691 0.99164345 0.96100279]
정확률(평균) = 97.219, 표준편차 = 0.015
```

위의 프로그램은 digit 데이터에 교차 검증을 적용한다. 
교차검증은 성능 측정 결과에 대한 신뢰도를 높일 목적으로 수행한다. k를 더 크게 하면, 매번 더 큰 훈련 집합을 사용할 뿐만 아니라 더 많은 수의 성능을 평균하므로 신뢰도가 높아진다. 대신 실행 시간이 더 걸리므로 상황에 따라 적절한 값을 설정해야 한다. 보통 5 또는 10을 사용한다.

## 3.7 인공지능은 어떻게 인식을 하나?

### 3.7.1 특징 공간을 분할하는 결정 경계
인공지능의 인식은 철저히 수학에 의존한다. 샘플은 특징 벡터로 표현되며, 특징 벡터는 특징 공간의 한 점에 해당한다. 인식 알고리즘은 원래 특징 공간을 인식 성능을 높이는 데 더 유리한 새로운 특징 공간으로 여러 단계에 걸쳐 변환하며, 최종적으로 특징 공간을 분할하여 구분해낸다. 요약하면, 인식 알고리즘은 특징 공간 변환과 특징 공간 분할로 분류 문제를 푼다.

***선형 분리 불가능(linearly non-separable)*** 이란 선형 모델, 즉 직선으로 100% 정확률을 달성할 수 없는 상황을 말한다. 원래 특징 공간에 식을 적용해 공간 변환을 시도하면 새로운 특징 공간이 되는데, 새로운 특징 공간에서는 선형 모델만 사용해도 100%의 정확률을 얻을 수 있다.

특징 공간을 분할하는 경계를 ***결정 경계(decision boundary)*** 라고 한다. 2차원 특징 공간의 결정 경계는 ***결정 직선(decision line)*** 이라고 하고, 3차원 특징 공간의 결정 경계는 ***결정 평면(decision plane)*** 이라고 한다.

딥러닝은 층이 매우 깊은 신경망 구조를 사용해 특징 공간 변환을 여러 단계에 걸쳐 수행하며 학습을 통해 최적화한다. 딥러닝이 높은 성능을 보장하는 이유 중의 하나는 보다 좋은 특징 공간을 찾아내는 능력 때문이다. 딥러닝에서는 좋은 특징을 찾는 일을 '***특징 학습(feature learning)*** 을 한다' 또는 '***표현 학습(representation learning)*** 을 한다.'고 말한다.

결정 경계를 정하는 문제에서는 몇 가지 사항을 고려해야 한다

1. 대부분의 데이터가 선형 분리 불가능하기 때문에 ***비선형 분류기(non-linear classifier)*** 가 필요하다. 이때는 결정 경계가 평면이 아니라 곡선 또는 곡면이어야 한다.
2. ***과잉 적합(overfitting)*** 을 피해야 한다. 학습 알고리즘이 아웃라이어(outlier)를 맞히려고 과다하게 복잡한 결정 경계를 만드는 상황을 과잉 적합이라고 한다. 과잉 적합이 발생하면 훈련 집합에 대한 정확률은 높지만 테스트 집합에 대한 정확률은 떨어지는 일반화 능력 저하 현상이 생긴다.

### 3.7.2 SVM의 원리

SVM 학습 알고리즘은 ***여백(margin)*** 을 최대화하는 결정 경계를 찾는다. 여백이란 결정경계와 훈련 집합애서 결정경계와 가장 가까운 샘플 까지의 거리를 말한다.

SVM은 선형 분류기에 불과하지만 선형 분류기의 한계를 벗아나기 위해 ***커널 트릭(kernel trick)*** 을 사용한다. 커널 트릭은 커널 함수를 사용해 선형 공간을 비선형 공간으로 확장한다. SVM에서 쓰는 커널 함수에는 polynomial, radial basis, sigmoid의 세 종류가 있다. 

```python
s = svm.SVC(gamma=0.1, C=10)
```
위의 라인에서 sklearn 라이브러리는 SVM을 위해 ***분류 모델 SVC*** 와 ***회귀 모델 SVR*** 을 제공하는데, 여기서는 분류 모델 SVC를 호출한다. 커널 함수를 지정하는 매개변수는 kernel인데 생략했기 때문에 기본값인 ***rbf(radial basis function)*** 를 사용한다. 매개변수 gamma는 커널 함수와 관련된 계수이다.

SVM에는 커널 함수 외에 또 다른 중요한 하이퍼 매개변수가 있다. 여백의 크기와 잘못 분류한 샘플의 수는 ***트레이드오프(trade-off)*** 관계에 있다. SVM에는 여백의 크기와 잘못 분류한 샘플의 수 사이에서 둘을 조정하는 C라는 하이퍼 매개변수가 있다. C를 크게 하면 잘못 분류한 샘플의 수는 적은데 여백이 작아지고, C를 작게 하면 여백은 큰데 잘못 분류한 샘플이 많아진다. 달리 말해 C를 크게 하면 훈련 집합에 대한 정확률은 높지만 일반화 능력은 떨어지고, C를 작게 하면 훈련 집합에 대한 정확률은 낮아지지만 일반화 능력은 커진다.