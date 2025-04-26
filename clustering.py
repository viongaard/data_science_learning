import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.data_loader import load_iris_data
from src.preprocessing import add_noise, add_missing_values, add_outliers, handle_missing_values, remove_outliers
from src.eda import plot_distributions, plot_correlations, plot_pairplot, plot_outliers
from src.modeling import train_kmeans
from src.evaluation import evaluate_clustering
from src.visualization import plot_pca_clustering


def main_clustering(method='kmeans'):
    # Создаём директорию для графиков
    os.makedirs('plots/clustering', exist_ok=True)

    # 1. Загрузка данных
    df = load_iris_data()
    print("Первые 5 строк:")
    print(df.head())
    print("\nИнформация о датасете:")
    print(df.info())
    print("\nОписательные статистики:")
    print(df.describe())

    # 2. Портим данные
    print("\nПропуски до имитации:")
    print(df.isnull().sum())
    df = add_noise(df, noise_level=0.1)
    df = add_outliers(df, outlier_ratio=0.05)
    df = add_missing_values(df, 'sepal length (cm)', fraction=0.2)
    print("\nПропуски после имитации:")
    print(df.isnull().sum())

    # 3. Предобработка
    df = handle_missing_values(df, method='median')
    print("\nПропуски после обработки:")
    print(df.isnull().sum())
    df = remove_outliers(df)
    print("\nРазмер после удаления выбросов:", df.shape)

    # 4. EDA
    plot_distributions(df, save_path='plots/clustering/distributions.png')
    plot_correlations(df, save_path='plots/clustering/correlations.png')
    plot_pairplot(df, save_path='plots/clustering/pairplot.png')
    plot_outliers(df, save_path='plots/clustering/outliers.png')

    # 5. Подготовка данных
    feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    X = df[feature_columns].to_numpy()

    # Нормализация (так как k-means чувствителен к масштабу данных)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Определение оптимального числа кластеров
    # Метод локтя
    inertia = []
    clusters_range = range(2, 11)
    for k in clusters_range:
        kmeans = train_kmeans(X_scaled, n_clusters=k)
        labels = kmeans.predict(X_scaled)
        centroids = kmeans.centroids.numpy()
        inertia_value = 0
        for i in range(len(X_scaled)):
            cluster_idx = labels[i]
            inertia_value += np.linalg.norm(X_scaled[i] - centroids[cluster_idx]) ** 2
        inertia.append(inertia_value)

    plt.figure(figsize=(6, 4))
    plt.plot(clusters_range, inertia, marker='o', linestyle='-', color='b')
    plt.xlabel("Количество кластеров")
    plt.ylabel("Inertia (сумма квадратов расстояний)")
    plt.title("Метод локтя")
    plt.savefig('plots/clustering/elbow_method.png')
    plt.close()

    # Метод силуэта
    silhouette_scores = []
    for k in clusters_range:
        kmeans = train_kmeans(X_scaled, n_clusters=k)
        labels = kmeans.predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, labels)
        silhouette_scores.append(silhouette_avg)

    plt.figure(figsize=(6, 4))
    plt.plot(clusters_range, silhouette_scores, marker='o', linestyle='-', color='g')
    plt.xlabel("Количество кластеров")
    plt.ylabel("Силуэтный коэффициент")
    plt.title("Метод силуэта")
    plt.savefig('plots/clustering/silhouette_method.png')
    plt.close()

    # 7. Кластеризация
    n_clusters = 3  # Выбираем 3 кластера (по результатам определения оптимального числа кластеров)
    kmeans = train_kmeans(X_scaled, n_clusters=n_clusters)
    labels = kmeans.predict(X_scaled)
    df['Cluster'] = labels

    # 8. Визуализация в PCA-пространстве
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plot_pca_clustering(X_pca, labels, save_path='plots/clustering/pca_kmeans.png')

    # 9. Оценка качества
    metrics = evaluate_clustering(X_scaled, labels)
    print("\nМетрики кластеризации:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main_clustering(method='kmeans')