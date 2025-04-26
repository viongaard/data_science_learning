import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src.data_loader import load_time_series_data
from src.preprocessing import handle_missing_values_ts, add_noise_ts
from src.eda import plot_time_series, plot_decomposition
from src.modeling import train_arima
from src.evaluation import evaluate_forecast
from src.visualization import plot_forecast


def main_time_series():
    # Создаём директорию для графиков
    import os
    os.makedirs("plots/time_series", exist_ok=True)

    # 1 Загрузка данных
    data = load_time_series_data('data/sales_data.csv', date_column='Month', value_column='Sales')
    print("Первые 5 строк:")
    print(data.head())
    print("\nИнформация о датасете:")
    print(data.info())

    # 2 Предобработка
    print("/nПропуски до имитации:")
    print(data.isnull().sum())
    data = add_noise_ts(data, noise_level=0.05)
    print("/nПропуски после имитации:")
    print(data.isnull().sum())
    data = handle_missing_values_ts(data, method='interpolate')
    print("\nПропуски после обработки:")
    print(data.isnull().sum())

    # 3 Визуализация исходного ряда
    plot_time_series(data, 'Sales', save_path='plots/time_series/sales.png')

    # 4 Анализ стационарности ряда
    print("\nПроверка стационарности (тест Дики-Фуллера):")
    result = adfuller(data['Sales'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    # Приведение к стационарности с помощью дифференцирования
    data['Sales_diff'] = data['Sales'].diff().dropna()
    data_diff = data.dropna()
    result = adfuller(data_diff['Sales_diff'])
    print('\nПосле дифференцирования:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    # 5 Декомпозиция временного ряда
    decomposition = seasonal_decompose(data['Sales'], model='additive', period=12)
    plot_decomposition(decomposition, save_path='plots/time_series/decomposition.png')

    # 6 Обучение модели и прогнозирование
    train_size = int(len(data) * 0.8)
    train, test = data['Sales'][:train_size], data['Sales'][train_size:]
    model = train_arima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

    # Прогноз на тестовую часть
    forecast = model.forecast(steps=len(test))
    forecast_index = test.index

    # Прогноз на будущее (следующие 12 месяцев)
    future_steps = 12
    future_forecast = model.forecast(steps=len(test) + future_steps)
    future_index = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='M')[1:]

    # 7. Оценка качества
    metrics = evaluate_forecast(test, forecast)
    print("\nМетрики прогнозирования:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 8. Визуализация результатов
    plot_forecast(data['Sales'], test, forecast, forecast_index, future_forecast[-future_steps:], future_index,
                  save_path='plots/time_series/forecast.png')


if __name__ == "__main__":
    main_time_series()