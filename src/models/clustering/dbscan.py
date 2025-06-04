import numpy as np
import torch
from typing import List, Set


class DBSCAN:
    """
    Реализация алгоритма DBSCAN
    """
    def __init__(self, eps=0.5, min_samples=5):
        """
        Конструктор

        :param eps: максимальное расстояние между двумя точками для их принадлежности к одному кластеру
        :param min_samples: минимальное количество точек для формирования плотной области
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = 0

    def _get_neighbors(self,
                       X_tensor: torch.Tensor,
                       point_idx: int) -> List[int]:
        """
        Находит всех соседей точки в радиусе eps

        Args:
            X_tensor: тензор с данными
            point_idx: индекс точки

        Returns:
            список индексов соседних точек
        """
        # Вычисляем расстояния от текущей точки до всех остальных
        distances = torch.norm(X_tensor - X_tensor[point_idx], dim=1)
        # Находим точки в радиусе eps
        neighbors = torch.where(distances <= self.eps)[0].tolist()
        return neighbors

    def _expand_cluster(self,
                        X_tensor: torch.Tensor,
                        point_idx: int,
                        neighbors: List[int],
                        cluster_id: int,
                        labels: np.ndarray,
                        visited: Set[int]) -> None:
        """
        Расширяет кластер, добавляя все точки, достижимые по плотности

        Args:
            X_tensor: тензор с данными
            point_idx: индекс начальной точки
            neighbors: список соседей начальной точки
            cluster_id: ID текущего кластера
            labels: массив меток кластеров
            visited: множество посещенных точек
        """
        # Помечаем текущую точку как принадлежащую кластеру
        labels[point_idx] = cluster_id

        # Обрабатываем всех соседей
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            # Если сосед еще не был посещен
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)

                # Находим соседей этого соседа
                neighbor_neighbors = self._get_neighbors(X_tensor, neighbor_idx)

                # Если у соседа достаточно соседей (он является кор-точкой)
                if len(neighbor_neighbors) >= self.min_samples:
                    # Добавляем его соседей к списку для обработки
                    for nn in neighbor_neighbors:
                        if nn not in neighbors:
                            neighbors.append(nn)

            # Если сосед еще не принадлежит никакому кластеру
            if labels[neighbor_idx] == -1:  # -1 означает шум
                labels[neighbor_idx] = cluster_id

            i += 1

    def fit(self, X):
        """
        Выполняет кластеризацию данных

        Args:
            X: массив данных для кластеризации
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        n_samples = X_tensor.shape[0]

        # Инициализируем метки (-1 означает шум)
        labels = np.full(n_samples, -1, dtype=int)
        visited = set()
        cluster_id = 0
        core_samples = []

        # Обрабатываем каждую точку
        for point_idx in range(n_samples):
            # Пропускаем уже посещенные точки
            if point_idx in visited:
                continue

            visited.add(point_idx)

            # Находим соседей текущей точки
            neighbors = self._get_neighbors(X_tensor, point_idx)

            # Если у точки недостаточно соседей, помечаем как шум
            if len(neighbors) < self.min_samples:
                continue

            # Точка является кор-точкой
            core_samples.append(point_idx)

            # Расширяем кластер из этой кор-точки
            self._expand_cluster(X_tensor, point_idx, neighbors, cluster_id, labels, visited)
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_samples)
        self.n_clusters_ = cluster_id

        return self

    def fit_predict(self, X):
        """
        Выполняет кластеризацию и возвращает метки

        Args:
            X: массив данных для кластеризации

        Returns:
            массив меток кластеров
        """
        self.fit(X)
        return self.labels_

    def get_params(self):
        """Возвращает параметры модели"""
        return {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'n_clusters': self.n_clusters_
        }