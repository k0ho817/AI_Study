#  iris 데이터의 분포를 특징 공간에 그리기
import plotly.express as px

df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length', color='species') # petal_length를 제외한 3차원 공간구성
fig.show(renderer="browser")