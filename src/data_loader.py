import kagglehub
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
import pandas as pd
import os


def load_iris_data():
    """
    Загружает датасет Iris и возвращает DataFrame.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

def load_breast_cancer_data():
    """
    Загружает датасет breast cancer и возвращает dataframe.
    """
    breast_cancer = load_breast_cancer()
    df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    df['target'] = breast_cancer.target
    return df

def load_groceries_data():
    """
    Загружает датасет 'Groceries_dataset.csv' с Kaggle через kagglehub и возвращает DataFrame.
    """
    path = kagglehub.dataset_download("heeraldedhia/groceries-dataset")
    csv_path = os.path.join(path, "Groceries_dataset.csv")
    df = pd.read_csv(csv_path)
    return df

def load_time_series_data(file_path, date_column, value_column):
    """
    Загружает датасет временных рядов из csv-файла.

    date_column указывает столбец с датами, который становится индексом DataFrame в формате datetime.
    Это позволяет работать с данными как с временным рядом.
    """
    df = pd.read_csv(file_path, parse_dates=[date_column], index_col=date_column)
    return df[[value_column]]
