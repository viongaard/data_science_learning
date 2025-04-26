import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y) # Создаём график

plt.title("ГОВНО БЛЯТЬ")
plt.xlabel("ось хуев")
plt.ylabel("ось уебанов")


import seaborn as sns

data = [1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8]
sns.histplot(data, kde=True)
plt.show()