import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(df, save_path=None):
    """Строит гистограммы для всех признаков."""
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(df.columns[:-1], 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[column], bins=30, kde=True)
        plt.title(f"Распределение {column}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_correlations(df, save_path=None):
    """Строит корреляционную матрицу."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Корреляционная матрица")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_pairplot(df, save_path=None):
    """Строит парные зависимости."""
    sns.pairplot(df, hue='target', diag_kind='hist')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_outliers(df, save_path=None):
    """Строит boxplot для проверки выбросов."""
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(df.columns[:-1], 1):
        plt.subplot(2, 2, i)
        sns.boxplot(y=df[column])
        plt.title(f"Boxplot {column}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
