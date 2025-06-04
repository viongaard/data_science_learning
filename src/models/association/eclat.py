from collections import defaultdict
from itertools import combinations

# Модель
class Eclat:
    # Конструктор
    def __init__(self, min_support=0.1):
        self.min_support = min_support
        self.transactions = []
        self.num_transactions = 0
        self.frequent_itemsets = {}

    # Запуск модели
    def fit(self, transactions):
        self.transactions = transactions
        self.num_transactions = len(transactions)
        vertical = defaultdict(set)

        # Построение вертикального представления
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                vertical[frozenset([item])].add(tid)

        def recursive(prefix, tids_prefix, items):
            for i in range(len(items)):
                item = items[i]
                tids_item = vertical[item]
                new_itemset = prefix | item
                new_tids = tids_prefix & tids_item if prefix else tids_item
                support = len(new_tids) / self.num_transactions
                if support >= self.min_support:
                    self.frequent_itemsets[new_itemset] = support
                    recursive(new_itemset, new_tids, items[i + 1:])

        singleton_items = list(vertical.keys())
        recursive(frozenset(), set(), singleton_items)
        return self.frequent_itemsets

    # Вычисление метрик
    def calculate_metrics(self, A, B):
        countA = sum(1 for t in self.transactions if A.issubset(t))
        countB = sum(1 for t in self.transactions if B.issubset(t))
        countAB = sum(1 for t in self.transactions if A.issubset(t) and B.issubset(t))

        supportA = countA / self.num_transactions
        supportB = countB / self.num_transactions
        supportAB = countAB / self.num_transactions

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
        itemsets = {k: v for k, v in self.frequent_itemsets.items() if len(k) >= min_len}
        for itemset in itemsets:
            for i in range(1, len(itemset)):
                for A in combinations(itemset, i):
                    A = frozenset(A)
                    B = itemset - A
                    if B:
                        metrics = self.calculate_metrics(A, B)
                        rules.append((A, B, metrics))
        return rules
