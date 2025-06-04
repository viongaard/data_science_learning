import numpy as np
import torch

# Класс реализации алгоритма k-means
class KMeans:
    def __init__(self, n_clusters=2, tol=1e-4):
        self.n_clusters = n_clusters
        self.tol = tol  # Погрешность

    def fit(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)  # Преобразуем в torch-тензор
        n_samples = X_tensor.shape[0]  # Кортеж размеров тензора по каждому измерению

        # 2 Выбираем сиды
        init_idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X_tensor[init_idxs]

        # 3 Для каждой записи определяем ближайший центр кластера
        while True:
            distances = torch.cdist(X_tensor, centroids)  # Вычисляем расстояния до кластеров
            labels = torch.argmin(distances, dim=1)  # Назначем метку кластера до которого расстояние минимально

            # 4 Обновляем центроиды
            new_centroids = torch.zeros_like(centroids)  # Создаём новый нулевой подобный тензор
            for k in range(self.n_clusters):
                cluster_points = X_tensor[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(dim=0)

            # 5 Проверяем критерий окончания - сходимость
            centroid_shift = torch.norm(new_centroids - centroids)
            if centroid_shift < self.tol:
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        distances = torch.cdist(X_tensor, self.centroids)
        return torch.argmin(distances, dim=1).numpy()