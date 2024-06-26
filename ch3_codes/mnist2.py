from sklearn import datasets
from sklearn import svm

digit = datasets.load_digits()

s = svm.SVC(gamma=0.1, C=10)
s.fit(digit.data, digit.target)

new_d = []