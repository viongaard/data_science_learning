from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.clustering.dbscan import DBSCAN
from src.models.clustering.k_means import KMeans

from src.models.classification.decision_tree_classifier import DecisionTreeClassifier
from src.models.classification.random_forest_classifier import RandomForestClassifier

from src.models.regression.linear_regression import LinearRegression


def prepare_data(df, target_column, feature_columns, scale=True):
    """Готовит данные для моделирования."""

    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_kmeans(X, n_clusters=2, tol=1e-4):
    model = KMeans(n_clusters=n_clusters, tol=tol)
    model.fit(X)
    return model


def train_dbscan(X, eps=0.5, min_samples=5):
    """
    Запускает DBSCAN

    :param X: множество точек
    :param eps: радиус окрестности точки
    :param min_samples: минимальное количество точек в окрестности для формирования кластера
    :return:
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model


def train_decision_tree_classifier(X_train, y_train, max_depth=5):
    """
    Обучает DecisionTreeClassifier
    """
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    return model


def train_random_forest_classifier(X_train, y_train, n_trees=15, max_depth=5):
    """Обучает RandomForestClassifier"""

    model = RandomForestClassifier(n_trees=n_trees, max_depth=max_depth)
    model.fit(X_train, y_train)

    return model


def train_linear_regression(X_train, y_train):
    """Обучает линейную регрессию."""

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model
