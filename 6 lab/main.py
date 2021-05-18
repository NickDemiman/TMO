import streamlit as st
import time
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.metrics import *
import matplotlib.pyplot as plt

def ENSEMBLES(models : list(), metrics : list(), x_test, y_test, size = 5, space = 0.2):
    columns = 1
    rows=len(metrics)
    index = 1
    fig = plt.figure(figsize=(size, (size + space)*rows))
    fig.subplots_adjust(hspace=space)
    pos = np.arange(len(models))

    for name in metrics:
        x = []
        y = []
        for func in models:
            data = func.predict(x_test)
            x.append(func.__class__.__name__)
            y.append(name(data, y_test))

        ax = fig.add_subplot(rows, columns, index)
        ax.barh(np.array(x), np.array(y), align='center')
        ax.set_title(name.__name__)
        for a,b in zip(pos, y):
            ax.text(0.1, a-0.1, str(round(b,3)), color='white')
        index+=1
    return fig

def load_class():
    data = fetch_covtype()
    df = pd.DataFrame(data=np.c_[data['data'], data['target']], columns=data['feature_names'] + ['target'])
    return data,df

def load_regress():
    data = fetch_california_housing()
    df = pd.DataFrame(data=np.c_[data['data'], data['target']], columns=data['feature_names'] + ['target'])
    return data,df


st.header('Вывод данных и графиков')
df = []
data = []
select = st.selectbox("Выберите вариант", ["Классификация", "Регрессия"])
st.header('Датафрейм')
if select == "Классификация":
    with st.spinner("Загрузка данных..."):
        data, df = load_class()
    st.subheader("dataset = cov_type")

elif select == "Регрессия":
    with st.spinner("Загрузка данных..."):
        data, df = load_regress()
    st.subheader("dataset = california_housing")

st.write(df.head())

test_size = st.sidebar.slider("Размер тестовой выборки", 0.1, 0.9, value=0.3,step=0.1)
n_estimators = st.sidebar.slider("Количество деревьев", 1, 15, value=5,step=1)
random_state = st.sidebar.slider("Число для инициализации гениратора случ. значений", 0,15,0,1)
models = st.sidebar.multiselect("Модель", ["Бэггинг","Бустинг", "Сверхслучайный лес", "Случайный лес", "Градиентный бустинг"])

if st.checkbox('Показать все данные'):
    st.subheader('Данные')
    st.write(data)

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=test_size, random_state=1)

# Модели
usable_models = []
if select == "Классификация":
    for key in models:
        if key=="Бэггинг":
            br = BaggingClassifier(n_estimators=n_estimators, random_state=random_state)
            br.fit(x_train, y_train)
            usable_models.append(br)

        if key=="Бустинг":
            adb = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
            adb.fit(x_train, y_train)
            usable_models.append(adb)  

        if key=="Сверхслучайный лес":
            ext = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state)
            ext.fit(x_train, y_train)
            usable_models.append(ext)

        if key=="Случайный лес":
            rfr = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            rfr.fit(x_train, y_train)
            usable_models.append(rfr)

        if key=="Градиентный бустинг":
            gbr = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
            gbr.fit(x_train, y_train)
            usable_models.append(gbr)

elif select == "Регрессия":
    for key in models:
        if key=="Бэггинг":
            print(1)
            br = BaggingRegressor(n_estimators=n_estimators, random_state=random_state)
            br.fit(x_train, y_train)
            usable_models.append(br)

        if key=="Бустинг":
            adb = AdaBoostRegressor(n_estimators=n_estimators, random_state=random_state)
            adb.fit(x_train, y_train)
            usable_models.append(adb)  

        if key=="Сверхслучайный лес":
            print
            ext = ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_state)
            ext.fit(x_train, y_train)
            usable_models.append(ext)

        if key=="Случайный лес":
            rfr = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
            rfr.fit(x_train, y_train)
            usable_models.append(rfr)

        if key=="Градиентный бустинг":
            gbr = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
            gbr.fit(x_train, y_train)
            usable_models.append(gbr)

# Метрики
metrics = [max_error, mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error]
metrics_names = [i.__name__ for i in metrics]

metrics_list = st.sidebar.multiselect("Выберите метрики", metrics_names)

# Данные для отображения metrics
Metrics = []

for i in metrics_list:
    for j in metrics:
        if i == j.__name__:
            Metrics.append(j)


fig = ENSEMBLES(usable_models, Metrics, x_test, y_test, 5, 0.4)
# plt.show()
st.pyplot(fig)