from collections import defaultdict


class Apriori:
    def __init__(self, min_support, transactions):
        self.min_support = min_support # Минимальная поддержка
        self.transactions = transactions # Список транзакций
        self.num_transactions = len(transactions) # Кол-во транзакций

    def execute(self):
        frequent_itemsets = {}  # Частые k-предметные наборы

        # 1. Находим частые 1-предметные наборы
        candidate_set = defaultdict(int) # Создаём словарь типа int с ключами
        for transaction in self.transactions:
            for item in transaction:
                candidate_set[frozenset([item])] += 1 # Подсчёт количества item в транзакциях

        # 2. Фильтруем по минимальной поддержке
        frequent_itemsets = {
            itemset: count / self.num_transactions # Вычисляем значение поддержки
            for itemset, count in candidate_set.items()
            if count / self.num_transactions >= self.min_support # Фильтруем по minsupport
        }

        # 3. Находим частые k-предметные наборы
        k = 2
        while frequent_itemsets:
            new_candidate_set = defaultdict(int)
            itemsets = list(frequent_itemsets.keys())

            # Генерация кандидатов размером k
            for i in range(len(itemsets)):
                for j in range(i + 1, len(itemsets)):
                    union = itemsets[i] | itemsets[j]
                    if len(union) == k:
                        new_candidate_set[union] = 0

            # Подсчёт поддержки кандидатов
            for transaction in self.transactions:
                for candidate in new_candidate_set:
                    if candidate.issubset(transaction):
                        new_candidate_set[candidate] += 1

            # Фильтрация по минимальной поддержке
            frequent_itemsets = {
                itemset: count / self.num_transactions
                for itemset, count in new_candidate_set.items()
                if count / self.num_transactions >= self.min_support
            }

            print(frequent_itemsets)
            k += 1

    def calculate_metrics(self, A, B):
        countA = sum(1 for transaction in self.transactions if A.issubset(transaction))
        countB = sum(1 for transaction in self.transactions if B.issubset(transaction))
        countAB = sum(1 for transaction in self.transactions if A.issubset(transaction) and B.issubset(transaction))

        supportA = countA / self.num_transactions
        supportB = countB / self.num_transactions
        supportAB = countAB / self.num_transactions

        lift = supportAB / (supportA * supportB) if supportA * supportB > 0 else 0
        leverage = supportAB - (supportA * supportB)

        print(f'Лифт: {lift}')
        print(f'Рычаг: {leverage}')


if __name__ == "__main__":
    transactions = [
        {'Молоко', 'Хлеб', 'Яйца'},
        {'Молоко', 'Масло', 'Кола'},
        {'Хлеб', 'Кола', 'Масло'},
        {'Молоко', 'Хлеб', 'Масло', 'Яйца'},
        {'Хлеб', 'Яйца', 'Масло'},
        {'Молоко', 'Хлеб', 'Яйца', 'Масло'},
        {'Кола', 'Масло'},
        {'Молоко', 'Масло'},
        {'Хлеб', 'Кола', 'Масло'},
        {'Молоко', 'Хлеб', 'Яйца', 'Кола'}
    ]

    apriori = Apriori(0.4, transactions)
    apriori.execute()

    A = {'Молоко'}
    B = {'Яйца'}
    apriori.calculate_metrics(A, B)
