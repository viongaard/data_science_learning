import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Реальные метки')
    plt.title('Матрица ошибок (Классификация)')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_pca_clustering(X_pca, labels, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k', s=75)
    plt.title('Результаты кластеризации K-Means (PCA 2D)')
    plt.xlabel('Первая главная компонента')
    plt.ylabel('Вторая главная компонента')
    plt.colorbar(label='Кластеры')
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_pca_classification(X, y_true, y_pred, save_path=None):
    """Визуализирует истинные и предсказанные классы с помощью PCA."""

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis')
    plt.title('Истинные классы')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis')
    plt.title('Предсказанные классы')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_predictions_regression(y_test, y_pred, save_path=None):
    """Строит график реальных и предсказанных значений."""

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Предсказания')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Идеальная линия')
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title('Реальные vs Предсказанные значения')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
