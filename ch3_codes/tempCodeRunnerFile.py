from sklearn import datasets
import matplotlib.pyplot as plt

news = datasets.fetch_20newsgroups(subset='train') # dataset read   
print(f"******\n{news.data[0]}\n******")
print(f"이 문서의 부류는 <{news.target_names[news.target[0]]}> 입니다.")