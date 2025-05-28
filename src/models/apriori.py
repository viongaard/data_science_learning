from collections import defaultdict
from itertools import combinations
import pandas as pd

# Модель
class Apriori:
    # Конструктор
    def __init__(self, min_support=0.1):
        self.min_support = min_support
        self.df = None
        self.num_transactions = 0
        self.frequent_itemsets = {}

    # Запуск модели
    def fit(self, df_binary: pd.DataFrame):
        self.df = df_binary
        self.num_transactions = len(df_binary)
        self.frequent_itemsets = {}

        # Находим частые 1-элементные наборы
        support_series = df_binary.sum() / self.num_transactions
        frequent_itemsets = {
            frozenset([item]): support
            for item, support in support_series.items()
            if support >= self.min_support
        }

        # Находим частые k-предметные наборы
        k = 2
        current_itemsets = list(frequent_itemsets.keys())
        all_frequent_itemsets = dict(frequent_itemsets)

        while current_itemsets:
            candidates = []
            seen = set()
            # Объединяем подмножества
            for i in range(len(current_itemsets)):
                for j in range(i + 1, len(current_itemsets)):
                    union = current_itemsets[i] | current_itemsets[j]
                    if len(union) == k and union not in seen:
                        seen.add(union)
                        candidates.append(union)

            # Отсекаем множества по поддержке
            new_frequent_itemsets = {}
            for candidate in candidates:
                cols = list(candidate)
                support = (df_binary[cols].all(axis=1).sum()) / self.num_transactions
                if support >= self.min_support:
                    new_frequent_itemsets[candidate] = support

            if not new_frequent_itemsets:
                break

            # Обновляем список частых наборов и увеличиваем k
            all_frequent_itemsets.update(new_frequent_itemsets)
            current_itemsets = list(new_frequent_itemsets.keys())
            k += 1

        # Возвращаем найденные частые наборы
        self.frequent_itemsets = all_frequent_itemsets
        return self.frequent_itemsets

    # Вычисление метрик
    def calculate_metrics(self, A, B):
        AB = list(A | B)
        A = list(A)
        B = list(B)

        supportA = (self.df[A].all(axis=1).sum()) / self.num_transactions
        supportB = (self.df[B].all(axis=1).sum()) / self.num_transactions
        supportAB = (self.df[AB].all(axis=1).sum()) / self.num_transactions

        lift = supportAB / (supportA * supportB) if supportA * supportB > 0 else 0
        leverage = supportAB - (supportA * supportB)
        confidence = supportAB / supportA if supportA > 0 else 0
        conviction = (1 - supportB) / (1 - confidence) if (1 - confidence) > 0 else float('inf')

        return {
            'support(A)': supportA,
            'support(B)': supportB,
            'support(AB)': supportAB,
            'confidence': confidence,
            'lift': lift,
            'leverage': leverage,
            'conviction': conviction
        }

    # Генерация правил
    def generate_rules(self, min_len=2):
        rules = []
        frequent_itemsets = {k: v for k, v in self.frequent_itemsets.items() if len(k) >= min_len}

        for itemset in frequent_itemsets:
            for i in range(1, len(itemset)):
                for A in combinations(itemset, i):
                    A = frozenset(A)
                    B = itemset - A
                    if len(B) == 0:
                        continue
                    metrics = self.calculate_metrics(A, B)
                    rules.append((A, B, metrics))

        return rules
