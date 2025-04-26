import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer  # KNNImputer - класс для заполнения пропущенных значений


def handle_missing_values(df, method='median'):
    """
    Обрабатывает пропуски в данных.
    """
    df_clean = df.copy()
    if method == 'median':
        # Выбираем столбцы, которые изначально числовые (даже если тип изменился из-за NaN)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for column in numeric_cols:
            if column != 'target':
                # Преобразуем столбец в числовой тип
                df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                # Вычисляем медиану (игнорируя NaN)
                median_value = df_clean[column].median()
                print(f"Столбец: {column}, Медиана: {median_value}, Пропуски до: {df_clean[column].isna().sum()}")
                # Заполняем пропуски
                df_clean[column] = df_clean[column].fillna(median_value)
                print(f"Пропуски после: {df_clean[column].isna().sum()}")
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'target']
        if numeric_cols:
            print(f"KNNImputer применяется к столбцам: {numeric_cols}")
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    return df_clean


def add_missing_values(df, column, fraction=0.2):
    """
    Заменяет часть данных пропусками.
    """
    df_with_missing = df.copy()
    np.random.seed(42)
    mask = np.random.choice([True, False], size=df.shape[0], p=[fraction, 1 - fraction])
    df_with_missing.loc[mask, column] = np.nan
    return df_with_missing


def add_noise(df, noise_level=0.1):
    """
    Добавляет шум к части данных
    """
    df_noisy = df.copy()
    # Выбираем числовые столбцы, исключая 'target'
    numeric_cols = [col for col in df_noisy.select_dtypes(include=['float64', 'int64']).columns if col != 'target']
    if not numeric_cols:
        return df_noisy
    # Создаём шум только для числовых столбцов
    noise = np.random.normal(0, noise_level, (df_noisy.shape[0], len(numeric_cols)))
    df_noisy[numeric_cols] += noise
    return df_noisy


def add_outliers(df, outlier_ratio=0.05):
    """
    Добавляет выбросы к данным.
    """
    df_outliers = df.copy()
    numeric_cols = df_outliers.select_dtypes(include=['float64', 'int64']).columns
    outlier_mask = np.random.random(df_outliers[numeric_cols].shape) < outlier_ratio
    df_outliers[numeric_cols] = df_outliers[numeric_cols].where(
        ~outlier_mask, df_outliers[numeric_cols] * np.random.uniform(5, 10, size=df_outliers[numeric_cols].shape)
    )
    return df_outliers


def remove_outliers(df):
    """Удаление выбросов с использованием метода IQR"""

    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    Q1 = df_clean[numeric_cols].quantile(0.25)
    Q3 = df_clean[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df_clean[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_clean[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

    return df_clean[mask]


def add_noise_ts(df, noise_level=0.1):
    df_noisy = df.copy()
    noise = np.random.normal(0, noise_level, df_noisy.shape[0])
    df_noisy.iloc[:, 0] += noise
    return df_noisy

def handle_missing_values_ts(df, method='interpolate'):
    df_filled = df.copy()
    if method == 'interpolate':
        df_filled = df_filled.interpolate(method='linear')
    elif method == 'mean':
        df_filled = df_filled.fillna(df_filled.mean())
    return df_filled
