# Прогнозирование 50-го перцентиля остатков на счетах клиентов
![Интерфейс приложения](inf.png)
## Описание проекта
Это приложение предсказывает 50-й перцентиль распределения суммарных остатков на накопительных счетах клиента на горизонте +2 месяца от отчетной даты. Модель использует ансамблевый подход (StackingRegressor) с комбинацией XGBoost и CatBoost. 

В преокте представлены:
- Ноутбук для отбора признаков
- Ноутбук для обучения модели
- Скрипт для JSON валидации 
- Тестовые файлы (row1.json and row2.json)

## Функциональность
- Загрузка JSON-файла с данными клиента
- Автоматическое создание синтетических признаков
- Предсказание 50-го перцентиля остатков
- Отображение результата в удобном интерфейсе

## Установка
1. Клонируйте репозиторий:
```bash
git clone https://github.com/zwaTOx/SHIFT_ML_07
cd SHIFT_ML_07```

2. Установите зависимости:
```bash
pip install pyproject.toml```

3. Запустите инференс модели:
```bash
streamlit run inference.py```



# Customer Account Balance 50th Percentile Prediction

## Project Description
![Application Interface](inf.png)

This application predicts the 50th percentile of total balance distribution across customer savings accounts for a 2-month forecast horizon. The model uses an ensemble approach (StackingRegressor) combining XGBoost and CatBoost algorithms.

The project includes:
- Feature selection scripts
- Model training files
- JSON validation utilities
- Test files (row1.json and row2.json)

## Features
- Customer data upload via JSON files
- Automated feature engineering
- 50th percentile balance prediction
- User-friendly result visualization

## Installation
1. Clone the repository:
```bash
git clone https://github.com/zwaTOx/SHIFT_ML_07
cd SHIFT_ML_07```

2. Install dependencies:
```bash
pip install pyproject.toml```

3. Run the inference application:
```bash
streamlit run inference.py```

