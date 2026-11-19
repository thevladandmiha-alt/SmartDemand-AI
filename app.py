import io
import pandas as pd
import streamlit as st
from core import FEATURES, demo_data, experiment, validate_data

st.set_page_config(page_title='SmartDemand AI', page_icon='📈', layout='wide')
st.title('SmartDemand AI')
st.caption('Учебная цифровая платформа: нейросетевое прогнозирование спроса и оценка точности')

@st.cache_data
def get_demo():
    return demo_data()

@st.cache_resource(show_spinner='Обучаем нейросеть…')
def cached_experiment(data_csv):
    return experiment(pd.read_csv(io.StringIO(data_csv)))

st.sidebar.header('Данные эксперимента')
source=st.sidebar.radio('Источник', ['Демонстрационные данные','Загрузить CSV'])
df=get_demo() if source=='Демонстрационные данные' else None
if source=='Загрузить CSV':
    upload=st.sidebar.file_uploader('CSV с колонками: '+', '.join(FEATURES+['Продажи']),type='csv')
    if upload:
        try:
            df=pd.read_csv(upload)
            df=validate_data(df)
        except Exception as e:
            st.error(f'Ошибка данных: {e}')
            st.stop()
    else:
        st.info('Загрузите CSV. Для образца можно скачать демонстрационный файл ниже.')
        st.sidebar.download_button('Скачать пример CSV', get_demo().to_csv(index=False).encode('utf-8-sig'),'пример_продаж.csv','text/csv')
        st.stop()

try:
    model, result, metrics=cached_experiment(df.to_csv(index=False))
except Exception as e:
    st.error(f'Не удалось провести эксперимент: {e}')
    st.stop()

t1,t2,t3,t4=st.tabs(['Прогноз и эксперимент','Сценарии закупки','Данные и экспорт','Архитектура платформы'])
with t1:
    a,b,c=st.columns(3)
    a.metric('MAE нейросети',f"{metrics['mae_model']:.2f} шт.")
    b.metric('MAE среднего',f"{metrics['mae_baseline']:.2f} шт.")
    c.metric('Изменение ошибки',f"{metrics['improvement_pct']:.1f}%")
    st.caption(f"Обучение: {metrics['train_count']} наблюдений; тест: {metrics['test_count']}. Разделение по времени, без перемешивания. MAE ниже — лучше.")
    st.line_chart(result[['Факт','Нейросеть','Среднее обучения']])
    st.bar_chart(pd.DataFrame({'MAE':[metrics['mae_model'],metrics['mae_baseline']]},index=['Нейросеть','Среднее']))
    st.info('Сравнение выполнено с простым постоянным прогнозом среднего спроса на обучающей выборке. Это не сравнение со всеми статистическими методами.')
with t2:
    c1,c2,c3,c4=st.columns(4)
    day=c1.selectbox('День недели',['Пн','Вт','Ср','Чт','Пт','Сб','Вс'],index=4)
    price=c2.number_input('Цена',min_value=0.0,max_value=1000000.0,value=100.0,step=5.0)
    discount=c3.slider('Скидка (%)',0,100,10)
    ads=c4.checkbox('Реклама активна',True)
    reserve=st.slider('Страховой запас (%)',0,50,15)
    current=pd.DataFrame([[['Пн','Вт','Ср','Чт','Пт','Сб','Вс'].index(day),price,discount,int(ads)]],columns=FEATURES)
    predicted=max(0,float(model.predict(current)[0]))
    suggested=int(__import__('math').ceil(predicted*(1+reserve/100)))
    c1,c2=st.columns(2)
    c1.metric('Прогноз спроса',f'{predicted:.1f} шт.')
    c2.metric('Ориентир закупки',f'{suggested} шт.')
    st.caption('Ориентир закупки = прогноз × (1 + страховой запас). Не учитывает остатки, сроки поставки и ограничения бюджета.')
    if source=='Демонстрационные данные':
        st.warning('Внимание: модель обучена на синтетических данных. Значения за пределами диапазона обучения (цена 80–120, скидка 0–20%) не следует интерпретировать как достоверные.')
with t3:
    st.dataframe(df.head(100),use_container_width=True)
    st.download_button('Скачать исходные данные CSV',df.to_csv(index=False).encode('utf-8-sig'),'smartdemand_data.csv','text/csv')
    st.download_button('Скачать прогнозы теста CSV',result.to_csv(index=False).encode('utf-8-sig'),'smartdemand_predictions.csv','text/csv')
    st.write('Обязательные поля:',', '.join(FEATURES+['Продажи']))
    st.caption('CSV должен содержать строки в хронологическом порядке. День недели: 0 = понедельник, 6 = воскресенье. Реклама: 0/1.')
with t4:
    st.markdown('''**Текущий прототип:** локальное Streamlit-приложение, Python, scikit-learn. Обучение и прогноз выполняются на том же компьютере, где запущено приложение.

**Предлагаемая облачная архитектура (пока не развёрнута):**

`Браузер → HTTPS → веб-приложение → API → сервис ML → хранилище данных и моделей`

- Контейнеризация приложения и сервиса модели (Docker)
- Объектное хранилище для CSV и обученных моделей
- Управляемая база данных для пользователей и истории запросов
- Мониторинг ошибок и времени ответа, контроль доступа
- Масштабирование сервиса прогнозов по нагрузке

**Математическое ядро:** стандартизация признаков, матричные операции, ReLU, оптимизация Adam, MAE на отложенной выборке.''')
    st.warning('Облачные компоненты представлены как проектное решение, а не как уже работающая инфраструктура.')
st.caption('SmartDemand AI · Учебный прототип · Результаты не являются коммерческими рекомендациями')
