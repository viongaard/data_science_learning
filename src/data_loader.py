from sklearn.datasets import load_iris
import pandas as pd


def load_iris_data():
    """
    Загружает датасет Iris и возвращает DataFrame.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df


def load_time_series_data(file_path, date_column, value_column):
    """
    Загружает датасет временных рядов из csv-файла.

    date_column указывает столбец с датами, который становится индексом DataFrame в формате datetime.
    Это позволяет работать с данными как с временным рядом.
    """
    df = pd.read_csv(file_path, parse_dates=[date_column], index_col=date_column)
    return df[[value_column]]
