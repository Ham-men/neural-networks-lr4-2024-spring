import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs  # Added import statement
from sklearn.cluster import KMeans

# Создание массивов данных
X1, y1 = make_blobs(n_samples=300, centers=[[2, 2], [-2, -2], [2, -2], [-2, 2], [0, 0]],
                    cluster_std=[0.3, 0.2, 0.3, 0.2, 0.3], random_state=42)

# Создание массивов данных
X2 = np.random.randn(100, 2)

# Создание массива данных в виде эллипсоида с добавлением шума
theta = np.linspace(0, 2*np.pi, 100)
r = 3 + np.random.randn(100)*0.1
X4 = np.vstack([r*np.cos(theta), 0.5*r*np.sin(theta)]).T

# Функция для кластеризации и визуализации результатов
def cluster_and_plot(X, title, ax):
    # Кластеризация методом K-средних
    kmeans = KMeans(n_clusters=5)#что это
    kmeans.fit(X)

    # Визуализация результатов
    ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
    ax.set_title(title)

# Кластеризация и визуализация результатов
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Corrected subplot creation
cluster_and_plot(X1, "Кластеризация групп точек", axs[0])  # Plotting on the first subplot
cluster_and_plot(X2, "Случайное распределение", axs[1])  # Plotting on the second subplot
plt.show()

# Визуализация второго набора данных
plt.figure(figsize=(7, 4))  # Create a new figure for the second plot
cluster_and_plot(X4, "Эллипс с применением шума", plt.gca())  # Plotting on the current axes
plt.title("Кластеризация методом K-средних")  # Setting the title for the second plot
plt.show()
