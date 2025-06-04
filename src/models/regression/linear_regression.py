import numpy as np


class LinearRegression:
    """Простая реализация линейной регрессии с использованием нормального уравнения"""

    def __init__(self):
        """Инициализирует модель"""

        self.weights = None

    def fit(self, X, y):
        """Обучает модель с использованием нормального уравнения"""

        # Добавляем столбец единиц для свободного члена (intercept)
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        XTX = np.dot(X_with_bias.T, X_with_bias)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(X_with_bias.T, y)
        self.weights = np.dot(XTX_inv, XTy)

        return self

    def predict(self, X):
        """Предсказывает значения для новых данных"""

        if self.weights is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit.")

        # Добавляем столбец единиц для свободного члена
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        # Предсказание y = Xw
        return np.dot(X_with_bias, self.weights)
