# AI_Study
파이썬으로 만드는 인공지능 요약

# Chapter01. 인간 지능을 흉내 내는 인공지능

## 1.2 인공지능을 바라보는 관점
### 1.2.1 인공지능에 대한 정의
> **지능** : 계산이나 문장 작성 따위의 지적 작업에서, 성취 정도에 따라 정하여지는 적응 능력

지적 작업에는 학습, 인식, 추론, 창작 등이 포함된다. <br>
학습은 경험이 쌓임에 따라 점점 성능이 좋아지는 지적 작업이고,<br>
인식은 오감을 통해 외부 환경의 상태를 알아내는 지적 작업이며,<br>
추론은 이미 알고 있는 사실을 바탕으로 새로운 사실 또는 새로운 지식을 알아내는 지적 작업이다.<br>
창작이란 세상에 없던 새로운 것을 만들어내는 지적 작업이다.<br>
또한 '적응'이란 인식이나 추론을 통해 알아낸 사실을 바탕으로 변화하는 환경에 자신을 맞추는 능력을 말한다. 

> **표준국어대사전의 인공지능에 대한 정의<br>**
**인공지능** : 인간의 지능이 가지는 학습, 추리, 적응, 논증 따위의 기능을 갖춘 컴퓨터 시스템

> **풀(Poole)이 정의한 인공지능 <br>**
**Artificial intelligence** : The field that studies the synthesis and analysis of computational agents that act initelligently<br>
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
```python
from sklearn import datasets # scikit-learn dataset module 불러오기

d = datasets.load_iris() # iris dataset 변수에 저장
print(d.DESCR) # iris dataset 정보 출력
```
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
위의 출력 결과를 보면 iris 데이터는 Setosam Versicolor, Virginina 세 부류로 구성되는 것을 알 수 있다.<br>
부류별로 50개씩 총 150개의 sample이 있고, 각 sample은 sepal length, sepal widath, petal length, petal width 4가지의 변수로 표현된다. 기계 학습에서는 각 변수를 ***특징(feature)*** 이라 부른다. 샘플 하나에 여러 개의 특징이 있으므로 이는 벡터로 표현되는데 이 벡터를 ***특징 벡터(feature vector)*** 라고 부른다.