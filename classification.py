import os

from src.data_loader import load_iris_data
from src.preprocessing import add_noise, add_missing_values, add_outliers, handle_missing_values, remove_outliers
from src.eda import plot_distributions, plot_correlations, plot_pairplot, plot_outliers
from src.modeling import prepare_data, train_decision_tree_classifier, train_random_forest_classifier
from src.evaluation import evaluate_classification
from src.visualization import plot_confusion_matrix, plot_pca_classification


def main_classification(method='decision_tree'):
    # Создаём директорию для графиков
    os.makedirs('plots/classification', exist_ok=True)

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

    # 3. Обрабатываем данные
    df = handle_missing_values(df, method='median')
    print("\nПропуски после обработки:")
    print(df.isnull().sum())
    df = remove_outliers(df)
    print("\nРазмер после удаления выбросов:", df.shape)

    # 4. EDA
    plot_distributions(df, save_path='plots/classification/distributions.png')
    plot_correlations(df, save_path='plots/classification/correlations.png')
    plot_pairplot(df, save_path='plots/classification/pairplot.png')
    plot_outliers(df, save_path='plots/classification/outliers.png')

    # 5. Подготовка данных
    feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    target_column = 'target'
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, feature_columns, scale=True)

    # 6. Обучение модели
    if method == 'decision_tree':
        model = train_decision_tree_classifier(X_train, y_train, max_depth=5)
    elif method == 'random_forest':
        model = train_random_forest_classifier(X_train, y_train, n_trees=10, max_depth=5)
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    # 7. Оценка
    metrics = evaluate_classification(model, X_test, y_test)
    print(f"\nМетрики классификации ({method}):")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 8. Визуализация
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, save_path=f'plots/classification/confusion_matrix_{method}.png')
    plot_pca_classification(X_test, y_test, y_pred, save_path=f'plots/classification/pca_{method}.png')


if __name__ == "__main__":
    # Тестируем оба метода
    main_classification(method='decision_tree')
    main_classification(method='random_forest')
