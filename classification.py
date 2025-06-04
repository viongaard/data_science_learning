import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.data_loader import load_iris_data
from src.preprocessing import add_noise, add_missing_values, add_outliers, handle_missing_values, remove_outliers
from src.eda import plot_distributions, plot_correlations, plot_pairplot, plot_outliers
from src.modeling import prepare_data, train_decision_tree_classifier, train_random_forest_classifier
from src.evaluation import evaluate_classification
from src.visualization import plot_confusion_matrix, plot_pca_classification, plot_predictions_regression


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


def main_classification(method='decision_tree'):
    # Создаём директорию для графиков
    os.makedirs('plots/classification', exist_ok=True)

    # 1. Загрузка данных
    df = load_iris_data()
    # print_data_info(df)

    # 2. Портим данные
    # print("\nПропуски до имитации:")
    # print(df.isnull().sum())
    df = add_noise(df, noise_level=0.1)
    df = add_outliers(df, outlier_ratio=0.05)
    df = add_missing_values(df, 'sepal length (cm)', fraction=0.2)
    # print("\nПропуски после имитации:")
    # print(df.isnull().sum())

    # 3. Обрабатываем данные
    df = handle_missing_values(df, method='median')
    # print("\nПропуски после обработки:")
    # print(df.isnull().sum())
    df = remove_outliers(df)
    # print("\nРазмер после удаления выбросов:", df.shape)

    # 4. EDA
    plot_distributions(df, save_path='plots/classification/distributions.png')
    plot_correlations(df, save_path='plots/classification/correlations.png')
    plot_pairplot(df, save_path='plots/classification/pairplot.png')
    plot_outliers(df, save_path='plots/classification/outliers.png')

    # 5. Подготовка данных
    feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    target_column = 'target'

    X = df[feature_columns]
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    # 6. Обучение модели
    if method == 'decision_tree':
        model = train_decision_tree_classifier(X_train, y_train, max_depth=6)
    elif method == 'random_forest':
        model = train_random_forest_classifier(X_train, y_train, n_trees=100, max_depth=6)
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


def main_regression_via_classification(method='decision_tree'):
    # Создание директории для графиков
    os.makedirs('plots/regression_via_classification', exist_ok=True)

    # Загрузка данных
    df = load_iris_data()

    # Минимальная обработка данных
    df = handle_missing_values(df, method='median')
    df = remove_outliers(df)

    # Подготовка данных
    # Для регрессии выбираем один из признаков в качестве целевой переменной
    feature_columns = ['sepal length (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
    target_column = 'sepal width (cm)'

    X = df[feature_columns]
    y = df[target_column]

    # Дискредитация целевой переменной на интервалы
    n_bins = 30  # Количество интервалов
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    y_binned = discretizer.fit_transform(y.values.reshape(-1, 1)).astype(int).flatten()

    # Разделение данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_binned, test_size=0.3, random_state=42)

    # 5. Обучение модели классификации
    if method == 'decision_tree':
        model = train_decision_tree_classifier(X_train, y_train, max_depth=10)
    elif method == 'random_forest':
        model = train_random_forest_classifier(X_train, y_train, n_trees=100, max_depth=6)
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    # 6. Предсказание классов
    y_pred_class = model.predict(X_test)

    # 7. Оценка качества классификации
    metrics = evaluate_classification(model, X_test, y_test)
    print(f"\nМетрики классификации ({method}):")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 8. Преобразование предсказанных классов обратно в числовые значения
    # Используем средние значения интервалов
    bin_edges = discretizer.bin_edges_[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    y_pred = bin_centers[y_pred_class.astype(int)]

    # 9. Оценка качества регрессии
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nМетрики регрессии:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    # 10. Визуализация
    plot_predictions_regression(y_test, y_pred,
                                save_path=f"plots/regression_via_classification/regression_{method}.png")
    plot_confusion_matrix(y_test, y_pred_class,
                          save_path=f'plots/regression_via_classification/confusion_matrix_{method}.png')
    plot_pca_classification(X_test, y_test, y_pred_class,
                            save_path=f'plots/regression_via_classification/pca_{method}.png')


if __name__ == "__main__":
    #main_classification(method='decision_tree')
    #main_classification(method='random_forest')
    main_regression_via_classification('decision_tree')
