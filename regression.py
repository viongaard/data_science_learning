import os
from src.data_loader import load_iris_data
from src.preprocessing import add_missing_values, handle_missing_values
from src.eda import plot_distributions, plot_correlations, plot_pairplot, plot_outliers
from src.modeling import prepare_data, train_linear_regression
from src.evaluation import evaluate_regression
from src.visualization import plot_predictions_regression


def main_regression():
    # Создаём директорию для графиков
    os.makedirs('plots/regression', exist_ok=True)

    # 1 Загрузка данных
    df = load_iris_data()
    print("Первые 5 строк:")
    print(df.head())
    print("\nИнформация о датасете:")
    print(df.info())
    print("\nОписательные статистики:")
    print(df.describe())

    # 2 Имитация пропусков (для тестирования)
    print("\nПропуски до имитации:")
    print(df.isnull().sum())
    df = add_missing_values(df, 'sepal length (cm)', fraction=0.2)
    print("\nПропуски после имитации:")
    print(df.isnull().sum())

    # 3 Обработка пропусков
    df = handle_missing_values(df, method='median')
    print("\nПропуски после обработки:")
    print(df.isnull().sum())

    # 4 EDA
    plot_distributions(df, save_path='plots/regression/distributions.png')
    plot_correlations(df, save_path='plots/regression/correlations.png')
    plot_pairplot(df, save_path='plots/regression/pairplot.png')
    plot_outliers(df, save_path='plots/regression/outliers.png')

    # 5 Подготовка данных
    feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']
    target_column = 'petal length (cm)'
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, feature_columns)

    # 6 Обучение модели
    model = train_linear_regression(X_train, y_train)

    # 7 Оценка
    metrics = evaluate_regression(model, X_test, y_test)
    print(f"\nМетрики модели: RMSE = {metrics['RMSE']:.4f}, R^2 = {metrics['R2']:.4f}")

    # 8 Визуализация предсказаний
    y_pred = model.predict(X_test)
    plot_predictions_regression(y_test, y_pred, save_path='plots/regression/prediction.png')


if __name__ == '__main__':
    main_regression()
