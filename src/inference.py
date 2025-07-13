import streamlit as st
import pandas as pd
import json
from json_validation import InputJSON
import pickle
import numpy as np

def create_new_features(df):
    # 1. Средний баланс за последние 3, 6, 12 месяцев
    df['avg_balance_3m'] = df[['avg_dep_avg_balance_1month_amt', 
                               'avg_dep_avg_balance_3month_amt']].mean(axis=1)
    df['avg_balance_6m'] = df[['avg_dep_avg_balance_3month_amt', 
                               'avg_dep_avg_balance_6month_amt']].mean(axis=1)
    df['avg_balance_12m'] = df[['avg_dep_avg_balance_6month_amt', 
                                'avg_dep_avg_balance_12month_amt']].mean(axis=1)
    # 2. Изменение баланса за последние месяцы
    df['balance_change_1m'] = df['avg_dep_avg_balance_1month_amt'] - df['avg_dep_avg_balance_3month_amt']
    df['balance_change_3m'] = df['avg_dep_avg_balance_3month_amt'] - df['avg_dep_avg_balance_6month_amt']
    df['balance_change_6m'] = df['avg_dep_avg_balance_6month_amt'] - df['avg_dep_avg_balance_12month_amt']
    # 3. Тренды по сбережениям
    df['savings_trend_3m'] = df['savings_sum_dep_now'] - df['savings_sum_dep_3m']
    df['savings_trend_6m'] = df['savings_sum_dep_now'] - df['savings_sum_dep_6m']
    # 4. Флаги крупных операций
    df['large_deposit_flag'] = np.where(df['max_max_dep_income_amt'] > df['avg_dep_avg_balance_12month_amt'], 1, 0)
    return df

# Функция для предсказания
def make_prediction(input_dict):
    #Считываем модель, порядок фичей и категориальные фичи, encoder из файла
    with open('src/stacking_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    model = artifacts['model']
    feature_order = artifacts['feature_order']
    encoder = artifacts['encoder']
    categorical_features = artifacts['categorical_features']
    df = pd.DataFrame([input_dict])
    df = df.fillna(0)
    # Добавляем синтетические признаки
    df = create_new_features(df)
    # Применяем OneHotEncoder
    encoded_features = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    # Удаляем исходные категориальные признаки
    df = df.drop(categorical_features, axis=1)
    # Объединяем с закодированными признаками
    df = pd.concat([df, encoded_df], axis=1)
    # Создаем DataFrame с правильным порядком признаков
    df = df[feature_order]
    # Делаем предсказание
    prediction = np.expm1(model.predict(df)[0])
    return prediction

# Интерфейс Streamlit
st.title('Прогнозирование 50-го перцентиля')
st.write('Это приложение предсказывает 50 перцентиль распределения суммарных остатков на всех накопительных счетах клиента на горизонте +2 мес. от отчетной даты')
# Загрузка JSON файла
uploaded_file = st.file_uploader("Загрузите JSON-файл с данными", type="json")
if uploaded_file is not None:
    data = json.load(uploaded_file)
    input_json = InputJSON(**data).model_dump()
    st.markdown(
        """
        <style>
        .stButton>button {
            float: right;
            margin-right: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button('Сделать предсказание'):
        prediction = make_prediction(input_json)
        st.subheader('Результат')
        st.success(f'**50-й перцентиль: {prediction:.4f}**')

    
