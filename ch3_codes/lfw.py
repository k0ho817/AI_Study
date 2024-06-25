#lfw 유명인 얼굴 데이터셋
from sklearn import datasets # 데이터셋 불러오기
import matplotlib.pyplot as plt # matplot으로 데이터 시각화

lfw = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4) #데이터 불러오기

plt.figure(figsize=(20,5)) # matplot 캔버스 크기 지정

for i in range(8):
    plt.subplot(1,8,i+1)
    plt.imshow(lfw.images[i],cmap=plt.cm.bone)
    plt.title(lfw.target_names[lfw.target[i]])

plt.show()