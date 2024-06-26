from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

digit = datasets.load_digits()
s = svm.SVC(gamma=0.001)
accuracies = cross_val_score(s, digit.data, digit.target, cv=5) # 5-겹 교차 검증

print(accuracies)
print(f'정확률(평균) = {accuracies.mean()*100:0.3f}, 표준편차 = {accuracies.std():0.3f}')
