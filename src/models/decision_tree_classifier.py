import torch
import numpy as np
import pandas as pd


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, probs=None):
        self.feature = feature  # Индекс признака для разделения
        self.threshold = threshold  # Порог для разделения
        self.left = left  # Левое поддерево
        self.right = right  # Правое поддерево
        self.value = value  # Значение (класс) для листа
        self.probs = probs  # Вероятности классов для листа


class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None

    def fit(self, X, y):
        """Обучает дерево решений"""

        # Преобразуем y в numpy.ndarray, если это pandas.Series
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # Преобразуем в тензоры
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.int64)

        self.n_classes = len(torch.unique(y_tensor))
        self.root = self._grow_tree(X_tensor, y_tensor, depth=0)

    def _gini(self, y):
        """Вычисляет коэффициент Джини"""

        _, counts = torch.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - torch.sum(probs ** 2)

    def _best_split(self, X, y):
        """Находит лучшее разделение для дерева"""

        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            thresholds = torch.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if torch.sum(left_mask) == 0 or torch.sum(right_mask) == 0:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                gini = (torch.sum(left_mask) * gini_left + torch.sum(right_mask) * gini_right) / n_samples

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold.item()

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth):
        """Собирает (растит) дерево решений"""

        n_samples = len(y)
        n_classes = len(torch.unique(y))

        # Считаем распределение классов
        class_counts = torch.bincount(y, minlength=self.n_classes)
        class_probs = class_counts / n_samples

        # Условия остановки
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            return Node(value=torch.argmax(class_counts).item(), probs=class_probs)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=torch.argmax(class_counts).item(), probs=class_probs)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature, threshold, left, right)

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return np.array([self._traverse_tree(x, self.root).probs.numpy() for x in X_tensor])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def _traverse_tree(self, x, node):
        if node.value is not None:  # В листе возвращаем узел, а не node.value
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
