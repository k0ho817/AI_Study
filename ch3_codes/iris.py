from sklearn import datasets

d = datasets.load_iris()
print(d.DESCR)

'''.. _iris_dataset:

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

해당 데이터셋은 sepal length, sepal width, petal length, petal width
4 종류의 특징값을 가지고 있다.
'''

for i in range(len(d.data)):
    print(i+1, d.data[i], d.target[i])

'''
데이터셋에 저장된 구조는 다음과 같다.
샘플의index [특징1, 특징2, 특징3, 특징4] 샘플의레이블
1 [5.1 3.5 1.4 0.2] 0
2 [4.9 3.  1.4 0.2] 0
3 [4.7 3.2 1.3 0.2] 0
4 [4.6 3.1 1.5 0.2] 0
5 [5.  3.6 1.4 0.2] 0
6 [5.4 3.9 1.7 0.4] 0
7 [4.6 3.4 1.4 0.3] 0
8 [5.  3.4 1.5 0.2] 0
9 [4.4 2.9 1.4 0.2] 0
10 [4.9 3.1 1.5 0.1] 0
11 [5.4 3.7 1.5 0.2] 0
12 [4.8 3.4 1.6 0.2] 0
13 [4.8 3.  1.4 0.1] 0
14 [4.3 3.  1.1 0.1] 0
15 [5.8 4.  1.2 0.2] 0
16 [5.7 4.4 1.5 0.4] 0
17 [5.4 3.9 1.3 0.4] 0
18 [5.1 3.5 1.4 0.3] 0
19 [5.7 3.8 1.7 0.3] 0
20 [5.1 3.8 1.5 0.3] 0
21 [5.4 3.4 1.7 0.2] 0
22 [5.1 3.7 1.5 0.4] 0
23 [4.6 3.6 1.  0.2] 0
24 [5.1 3.3 1.7 0.5] 0
25 [4.8 3.4 1.9 0.2] 0
26 [5.  3.  1.6 0.2] 0
27 [5.  3.4 1.6 0.4] 0
28 [5.2 3.5 1.5 0.2] 0
29 [5.2 3.4 1.4 0.2] 0
30 [4.7 3.2 1.6 0.2] 0
31 [4.8 3.1 1.6 0.2] 0
32 [5.4 3.4 1.5 0.4] 0
33 [5.2 4.1 1.5 0.1] 0
34 [5.5 4.2 1.4 0.2] 0
35 [4.9 3.1 1.5 0.2] 0
36 [5.  3.2 1.2 0.2] 0
37 [5.5 3.5 1.3 0.2] 0
38 [4.9 3.6 1.4 0.1] 0
39 [4.4 3.  1.3 0.2] 0
40 [5.1 3.4 1.5 0.2] 0
41 [5.  3.5 1.3 0.3] 0
42 [4.5 2.3 1.3 0.3] 0
43 [4.4 3.2 1.3 0.2] 0
44 [5.  3.5 1.6 0.6] 0
45 [5.1 3.8 1.9 0.4] 0
46 [4.8 3.  1.4 0.3] 0
47 [5.1 3.8 1.6 0.2] 0
48 [4.6 3.2 1.4 0.2] 0
49 [5.3 3.7 1.5 0.2] 0
50 [5.  3.3 1.4 0.2] 0
51 [7.  3.2 4.7 1.4] 1
52 [6.4 3.2 4.5 1.5] 1
53 [6.9 3.1 4.9 1.5] 1
54 [5.5 2.3 4.  1.3] 1
55 [6.5 2.8 4.6 1.5] 1
56 [5.7 2.8 4.5 1.3] 1
57 [6.3 3.3 4.7 1.6] 1
58 [4.9 2.4 3.3 1. ] 1
59 [6.6 2.9 4.6 1.3] 1
60 [5.2 2.7 3.9 1.4] 1
61 [5.  2.  3.5 1. ] 1
62 [5.9 3.  4.2 1.5] 1
63 [6.  2.2 4.  1. ] 1
64 [6.1 2.9 4.7 1.4] 1
65 [5.6 2.9 3.6 1.3] 1
66 [6.7 3.1 4.4 1.4] 1
67 [5.6 3.  4.5 1.5] 1
68 [5.8 2.7 4.1 1. ] 1
69 [6.2 2.2 4.5 1.5] 1
70 [5.6 2.5 3.9 1.1] 1
71 [5.9 3.2 4.8 1.8] 1
72 [6.1 2.8 4.  1.3] 1
73 [6.3 2.5 4.9 1.5] 1
74 [6.1 2.8 4.7 1.2] 1
75 [6.4 2.9 4.3 1.3] 1
76 [6.6 3.  4.4 1.4] 1
77 [6.8 2.8 4.8 1.4] 1
78 [6.7 3.  5.  1.7] 1
79 [6.  2.9 4.5 1.5] 1
80 [5.7 2.6 3.5 1. ] 1
81 [5.5 2.4 3.8 1.1] 1
82 [5.5 2.4 3.7 1. ] 1
83 [5.8 2.7 3.9 1.2] 1
84 [6.  2.7 5.1 1.6] 1
85 [5.4 3.  4.5 1.5] 1
86 [6.  3.4 4.5 1.6] 1
87 [6.7 3.1 4.7 1.5] 1
88 [6.3 2.3 4.4 1.3] 1
89 [5.6 3.  4.1 1.3] 1
90 [5.5 2.5 4.  1.3] 1
91 [5.5 2.6 4.4 1.2] 1
92 [6.1 3.  4.6 1.4] 1
93 [5.8 2.6 4.  1.2] 1
94 [5.  2.3 3.3 1. ] 1
95 [5.6 2.7 4.2 1.3] 1
96 [5.7 3.  4.2 1.2] 1
97 [5.7 2.9 4.2 1.3] 1
98 [6.2 2.9 4.3 1.3] 1
99 [5.1 2.5 3.  1.1] 1
100 [5.7 2.8 4.1 1.3] 1
101 [6.3 3.3 6.  2.5] 2
102 [5.8 2.7 5.1 1.9] 2
103 [7.1 3.  5.9 2.1] 2
104 [6.3 2.9 5.6 1.8] 2
105 [6.5 3.  5.8 2.2] 2
106 [7.6 3.  6.6 2.1] 2
107 [4.9 2.5 4.5 1.7] 2
108 [7.3 2.9 6.3 1.8] 2
109 [6.7 2.5 5.8 1.8] 2
110 [7.2 3.6 6.1 2.5] 2
111 [6.5 3.2 5.1 2. ] 2
112 [6.4 2.7 5.3 1.9] 2
113 [6.8 3.  5.5 2.1] 2
114 [5.7 2.5 5.  2. ] 2
115 [5.8 2.8 5.1 2.4] 2
116 [6.4 3.2 5.3 2.3] 2
117 [6.5 3.  5.5 1.8] 2
118 [7.7 3.8 6.7 2.2] 2
119 [7.7 2.6 6.9 2.3] 2
120 [6.  2.2 5.  1.5] 2
121 [6.9 3.2 5.7 2.3] 2
122 [5.6 2.8 4.9 2. ] 2
123 [7.7 2.8 6.7 2. ] 2
124 [6.3 2.7 4.9 1.8] 2
125 [6.7 3.3 5.7 2.1] 2
126 [7.2 3.2 6.  1.8] 2
127 [6.2 2.8 4.8 1.8] 2
128 [6.1 3.  4.9 1.8] 2
129 [6.4 2.8 5.6 2.1] 2
130 [7.2 3.  5.8 1.6] 2
131 [7.4 2.8 6.1 1.9] 2
132 [7.9 3.8 6.4 2. ] 2
133 [6.4 2.8 5.6 2.2] 2
134 [6.3 2.8 5.1 1.5] 2
135 [6.1 2.6 5.6 1.4] 2
136 [7.7 3.  6.1 2.3] 2
137 [6.3 3.4 5.6 2.4] 2
138 [6.4 3.1 5.5 1.8] 2
139 [6.  3.  4.8 1.8] 2
140 [6.9 3.1 5.4 2.1] 2
141 [6.7 3.1 5.6 2.4] 2
142 [6.9 3.1 5.1 2.3] 2
143 [5.8 2.7 5.1 1.9] 2
144 [6.8 3.2 5.9 2.3] 2
145 [6.7 3.3 5.7 2.5] 2
146 [6.7 3.  5.2 2.3] 2
147 [6.3 2.5 5.  1.9] 2
148 [6.5 3.  5.2 2. ] 2
149 [6.2 3.4 5.4 2.3] 2
150 [5.9 3.  5.1 1.8] 2

s개의 d개의 특징을 가진 데이터를 표현하는 방법은 다음과 같다.
특징 벡터 : x_1 = (x_1, x_2, x_3, ... x_d),
            x_2 = (x_1, x_2, x_3, ... x_d),
            x_3 = (x_1, x_2, x_3, ... x_d),
            ...
            x_s = (x_1, x_2, x_3, ... x_d)
d개의 특징을 가진다면 벡터의 차원은 d로 설정된다.

부류의 갯수 c로 설정된 샘플의 레이블 y는 0, 1, 2, ... c-1 중의 한 값을 갖게 된다.
기계학습은 때때로 원핫 코드 one-hot code로 y를 표현하기도 한다.
원핫코드는 요소 하나만 1인 이진 벡터이다. (디코더와 같은 형식)
ex) 0 = (1,0,0) 1 = (0,1,0) 2 = (0,0,1)
때때로 다중레이블도 허용한
ex) 고양이의 레이블값 : 2
    쥐의 레이블값 : 5
    고양이와 쥐가 같이 나온사진의 레이블값 (0,0,1,0,0,1,0,0,...)


'''

from sklearn import svm

s = svm.SVC(gamma=0.1, C=10) #svm 분류 모델 SVC 객체 생성 (param 뒤에서 나옴)
s.fit(d.data, d.target) #iris 데이터로 학습
                        #fit 함수의 훈련집합 : iris
                        #param1 : 특징벡터 x_1,x_2,x_3...x_d
                        #param2 : 샘플의 레이블 y = 0,1,2,...c-1

new_d = [[6.4, 3.2, 6.0, 2.5], [7.1, 3.1, 4.7, 1.35]] #새로운 객체 2개 생성

res = s.predict(new_d) #객체 분류
print("새로운 2개 샘플의 부류는 ", res)

'''
데이터 편향 : 데이터를 다양하게 수집하지 않고 특정 특징에 편향되게 수집하는 경우 정확률에 부정적인 요인이 발생한다.
모델링 : 기계학습에서 규칙을 만드는 일
기계 학습의 데이터를 만드는 과정, 즉 특징 벡터를 추출하고 레이블을 붙이는 과정은 규칙 기반과 같다. 하지만 규칙을 만드는 일은 기계 학습 모델을 이용해 자동으로 수행한다.
딥러닝에서는 레이블은 고전적 기계 학습과 마찬가지로 전문가의 손을 거쳐 만들지만 특징 벡터는 학습이 자동으로 알아낸다.
'''