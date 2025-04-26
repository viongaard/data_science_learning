import numpy as np

from src.models.decision_tree_classifier import DecisionTreeClassifier


# Реализация случайного леса
class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.to_numpy()
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_trees):
            # Bootstrap выборка
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Случайный выбор признаков
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_sample = X_sample[:, feature_indices]

            # Обучаем дерево
            tree = DecisionTreeClassifier(self.max_depth, self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, feature_indices))

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], self.trees[0][0].n_classes))
        for tree, feature_indices in self.trees:
            probs += tree.predict_proba(X[:, feature_indices])
        return probs / self.n_trees

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
