import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

train = pd.read_parquet('data/train/train_main_df.parquet')
target = pd.read_csv('data/info/train_target.csv')
test = pd.read_parquet('data/test/test_main_df.parquet')

# Выбираем только нужные признаки
features = ['customer_age', 'savings_sum_dep_now']
X_train = train[features]
y_train = np.log1p(target['target'])  # Логарифмируем таргет как в исходном коде
X_test = test[features]

# Обучаем линейную регрессию
model = LinearRegression()
model.fit(X_train, y_train)

# Делаем предсказания
test_predict = model.predict(X_test)
test_full_predict = np.exp(test_predict) - 1  # Обратное преобразование из логарифма

# Формируем сабмит
submission = pd.DataFrame()
submission['id'] = test['id']
submission['target'] = test_full_predict
submission.to_csv('data/submission/submission.csv', index=False)