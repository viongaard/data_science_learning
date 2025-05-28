from collections import defaultdict, OrderedDict
from itertools import combinations


# Узел дерева
class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None  # Ссылка на следующий узел с тем же item


# Дерево
class FPTree:
    # Конструктор
    def __init__(self, transactions, min_support):
        self.min_support = min_support
        self.headers = defaultdict(list)
        self.root = FPNode(None, 0, None)
        self._build_tree(transactions)

    # Построение дерева
    def _build_tree(self, transactions):
        # Первый проход: подсчет частых 1-предметных наборов
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        # Фильтрация по min_support и сортировка
        self.frequent_items = {item: count for item, count in item_counts.items()
                               if count >= self.min_support}
        frequent_items_sorted = sorted(self.frequent_items.keys(),
                                       key=lambda x: (-self.frequent_items[x], x))

        # Второй проход: построение дерева
        for transaction in transactions:
            # Фильтруем по частоте и сортируем наборы по убыванию
            filtered = [item for item in transaction if item in self.frequent_items]
            filtered.sort(key=lambda x: (-self.frequent_items[x], x))

            # Добавляем в дерево узлы
            current_node = self.root
            for item in filtered:
                if item in current_node.children:
                    child = current_node.children[item]
                    child.count += 1
                else:
                    child = FPNode(item, 1, current_node)
                    current_node.children[item] = child
                    # Обновляем header table
                    self.headers[item].append(child)
                current_node = child


    # Получение условной базы паттернов
    def _get_conditional_pattern_base(self, item):
        patterns = []
        for node in self.headers[item]:
            prefix_path = []
            current = node.parent
            while current.item is not None:
                prefix_path.append(current.item)
                current = current.parent
            if prefix_path:
                patterns.append((prefix_path, node.count))
        return patterns

    # Рекурсивная добыча частых наборов
    def _mine_tree(self, suffix, frequent_itemsets):
        items = [item for item in self.headers.keys()]
        items.sort(key=lambda x: (-self.frequent_items[x], x))

        for item in items:
            new_suffix = [item] + suffix
            frequent_itemsets[frozenset(new_suffix)] = self.frequent_items[item]

            # Строим условное FP-дерево
            patterns = self._get_conditional_pattern_base(item)
            conditional_transactions = []
            for pattern, count in patterns:
                conditional_transactions.extend([pattern] * count)

            if conditional_transactions:
                conditional_tree = FPTree(conditional_transactions, self.min_support)
                if conditional_tree.headers:
                    conditional_tree._mine_tree(new_suffix, frequent_itemsets)


# Модель
class FPGrowth:
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

        # Конвертируем абсолютный min_support в минимальное количество
        abs_min_support = self.min_support * self.num_transactions

        # Строим FP-дерево
        tree = FPTree(transactions, abs_min_support)

        # Добываем частые наборы
        self.frequent_itemsets = {}
        tree._mine_tree([], self.frequent_itemsets)

        # Конвертируем counts в поддержку
        self.frequent_itemsets = {
            itemset: count / self.num_transactions
            for itemset, count in self.frequent_itemsets.items()
        }

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