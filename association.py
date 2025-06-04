import time
import tracemalloc
import pandas as pd
from src.data_loader import load_groceries_data
from src.models.association.apriori import Apriori
from src.models.association.fpgrowth import FPGrowth
from src.models.association.eclat import Eclat

# Функция перевода списка транзакций в бинарную матрицу (для априори)
def transactions_to_binary_df(transactions):
    all_items = sorted(set().union(*transactions))
    df = pd.DataFrame(0, index=range(len(transactions)), columns=all_items)
    for idx, transaction in enumerate(transactions):
        df.loc[idx, list(transaction)] = 1
    return df

# Функция общей оценки результатов алгоритма
def evaluate_rules(rules):
    top_rules = sorted(rules, key=lambda x: -x[2]['confidence'])[:10]
    avg_conf = sum(r[2]['confidence'] for r in top_rules) / len(top_rules)
    avg_lift = sum(r[2]['lift'] for r in top_rules) / len(top_rules)
    return {
        'total_rules': len(rules),
        'avg_confidence': avg_conf,
        'avg_lift': avg_lift
    }

# Функция запуска алгоритма
def run_algorithm(name, transactions, binary_df):
    print(f"\n{'='*20} {name} {'='*20}")
    if name == 'Apriori':
        model = Apriori(min_support=0.01)
        data = binary_df
    elif name == 'FPGrowth':
        model = FPGrowth(min_support=0.01)
        data = transactions
    elif name == 'Eclat':
        model = Eclat(min_support=0.01)
        data = transactions
    else:
        raise ValueError("Unknown algorithm")

    # Измеряем время и память
    tracemalloc.start()
    start_time = time.time()

    frequent_itemsets = model.fit(data)
    rules = model.generate_rules(min_len=2)

    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = evaluate_rules(rules)

    print(f"Время выполнения: {elapsed_time:.4f} сек")
    print(f"Использование памяти: {peak / 1024:.2f} КБ")
    print(f"Найдено правил: {metrics['total_rules']}")
    print(f"Средняя уверенность топ-10: {metrics['avg_confidence']:.3f}")
    print(f"Средний lift топ-10: {metrics['avg_lift']:.3f}")
    print("Примеры правил:")
    for A, B, m in sorted(rules, key=lambda x: -x[2]['confidence'])[:5]:
        print(f"{set(A)} => {set(B)}, conf = {m['confidence']:.2f}, lift = {m['lift']:.2f}")

# Основная функция
def main_association():
    df = load_groceries_data()
    grouped = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(set)
    transactions = list(grouped)
    binary_df = transactions_to_binary_df(transactions)

    for method in ['Apriori', 'FPGrowth', 'Eclat']:
        run_algorithm(method, transactions, binary_df)

if __name__ == '__main__':
    main_association()
