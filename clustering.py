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
from src.modeling import train_kmeans, train_dbscan
from src.evaluation import evaluate_clustering
from src.visualization import plot_pca_clustering


def print_data_info(df, title="Основная информация о датафрейме"):
    """
    Выводит основную информацию о датафрейме

    :param df: датафрейм
    :param title: заголовок
    """
    print(f"\n{'=' * 20} {title} {'=' * 20}")

    print("\nПервые 5 строк:")
    print(df.head())

    print("\nИнформация о датасете:")
    print(df.info())

    print("\nОписательные статистики:")
    print(df.describe())

    print("\nКоличество пропусков по столбцам:")
    print(df.isnull().sum())


def corrupt_data(df, noise_level=0.1, outlier_ratio=0.05, missing_values_config=None):
    """
    Добавляет шум, выбросы и пропуски в данные

    :param df: датафрейм
    :param noise_level: уровень шума (0-1)
    :param outlier_ratio: доля выбросов (0-1)
    :param missing_values_config: конфигурация для пропусков вида {'column_name': fraction, ...}
    :return: модифицированный датафрейм
    """
    print("\nПропуски до имитации:")
    print(df.isnull().sum())

    # Добавляем шум
    if noise_level > 0:
        df = add_noise(df, noise_level=noise_level)

    # Добавляем выбросы
    if outlier_ratio > 0:
        df = add_outliers(df, outlier_ratio=outlier_ratio)

    # Добавляем пропуски по конфигурации
    if missing_values_config:
        for col, fraction in missing_values_config.items():
            if col in df.columns:
                df = add_missing_values(df, col, fraction=fraction)

    print("\nПропуски после имитации:")
    print(df.isnull().sum())

    return df


def visualize_eda(df):
    """
    Визуализирует данные для предиктивной аналитики

    :param df: датафрейм
    """
    plot_distributions(df, save_path='plots/clustering/distributions.png')
    plot_correlations(df, save_path='plots/clustering/correlations.png')
    plot_pairplot(df, save_path='plots/clustering/pairplot.png')
    plot_outliers(df, save_path='plots/clustering/outliers.png')


def run_elbow_method(X_scaled, clusters_range):
    """
    Строит график инерции для нахождения оптимального количества кластеров

    :param X_scaled:
    :param clusters_range:
    :return:
    """
    inertia = []
    for k in clusters_range:
        kmeans = train_kmeans(X_scaled, n_clusters=k)
        labels = kmeans.predict(X_scaled)
        centroids = kmeans.centroids.numpy()
        inertia_value = sum(np.linalg.norm(X_scaled[i] - centroids[labels[i]]) ** 2 for i in range(len(X_scaled)))
        inertia.append(inertia_value)

    plt.figure(figsize=(6, 4))
    plt.plot(clusters_range, inertia, marker='o', linestyle='-', color='b')
    plt.xlabel("Количество кластеров")
    plt.ylabel("Inertia (сумма квадратов расстояний)")
    plt.title("Метод локтя")
    plt.savefig('plots/clustering/elbow_method.png')
    plt.close()


def run_sulhouette_method(X_scaled, clusters_range):
    """
    Стоит график силуэта для нахождения оптимального количества кластеров
    :param X_scaled:
    :param clusters_range:
    :return:
    """
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


def apply_pca_and_plot(X_scaled, labels, path):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plot_pca_clustering(X_pca, labels, save_path=path)


def run_algorithm(name, df):
    """
    Запускает выбранный алгоритм

    :param name: название алгоритма
    :param df: датафрейм
    :return: результат работы алгоритма
    """
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    if name == 'KMeans':
        return train_kmeans(df, n_clusters=3, tol=1e-4)
    elif name == 'DBSCAN':
        return train_dbscan(df, eps=0.3, min_samples=5)
    else:
        raise ValueError("Unknown algorithm")


def main_clustering():
    os.makedirs('plots/clustering', exist_ok=True)

    # === 1. Загрузка и первичная информация ===
    df = load_iris_data()
    print_data_info(df, title="Исходные данные: Ирисы Фишера")

    # === 2. Искажение данных: шум, выбросы, пропуски ===
    corrupt_data(df, noise_level=0.1, outlier_ratio=0.05, missing_values_config={
        'sepal length (cm)': 0.2,
        'petal width (cm)': 0.1
    })

    # === 3. Предобработка: пропуски, выбросы, масштабирование ===
    df = handle_missing_values(df, method='median')
    df = remove_outliers(df)

    feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    X = df[feature_columns].to_numpy()
    X_scaled = MinMaxScaler().fit_transform(X)

    # === 4. Визуализация и оценка числа кластеров ===
    visualize_eda(df)
    clusters_range = range(2, 11)
    run_elbow_method(X_scaled, clusters_range)
    run_sulhouette_method(X_scaled, clusters_range)

    # === 5. Кластеризация двумя методами: KMeans и DBSCAN ===
    algorithms = ['KMeans', 'DBSCAN']
    for algo_name in algorithms:
        model = run_algorithm(algo_name, X_scaled)

        if hasattr(model, 'predict'):
            labels = model.predict(X_scaled)
        else:
            labels = model.labels_

        df[f'Cluster_{algo_name}'] = labels

        # Визуализация кластеров в PCA
        pca_path = f'plots/clustering/pca_{algo_name.lower()}.png'
        apply_pca_and_plot(X_scaled, labels, path=pca_path)

        # Оценка качества кластеризации
        print(f"\nМетрики для {algo_name}:")
        try:
            metrics = evaluate_clustering(X_scaled, labels)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        except Exception as e:
            print(f"Ошибка при вычислении метрик для {algo_name}: {e}")

    # === 6. Интерпретация кластеров\ ===
    print("\n=== Интерпретация кластеров ===")
    for algo_name in algorithms:
        print(f"\nАлгоритм: {algo_name}")
        print(df.groupby(f'Cluster_{algo_name}')[feature_columns].mean())


if __name__ == "__main__":
    main_clustering()
