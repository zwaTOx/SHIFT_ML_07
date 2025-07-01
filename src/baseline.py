import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Загружаем данные
train = pd.read_parquet('data/train/train_main_df.parquet')
target = pd.read_csv('data/info/train_target.csv')
test = pd.read_parquet('data/test/test_main_df.parquet')

# Выбираем только нужные признаки
features = ['customer_age', 'savings_sum_dep_now']
X_train = train[features]
y_train = np.log1p(target['target'])
X_test = test[features]

# Делим данные на обучающую и валидационную выборки
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X_train_split, y_train_split)

# Делаем предсказания на валидационной и тестовой выборке
val_predict = model.predict(X_val)
test_predict = model.predict(X_test)  

# Обратное преобразование для валидации
val_predict_exp = np.exp(val_predict) - 1
y_val_exp = np.exp(y_val) - 1

# Вычисляем RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val_exp, val_predict_exp))
print(f'RMSLE: {rmsle}')

# Формируем сабмит
submission = pd.DataFrame()
submission['id'] = test['id'].values
submission['target'] = np.exp(test_predict) - 1  
submission.to_csv('data/submission/submission.csv', index=False)