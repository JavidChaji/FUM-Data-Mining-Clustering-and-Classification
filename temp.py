from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import pandas as pd


dataset = np.load('./extracted_vectors.npy')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(dataset)

X_normalized = normalize(X_scaled)

X_normalized = pd.DataFrame(X_normalized)

pca = PCA()
dataset = pca.fit_transform(X_normalized)
print(dataset)


neighbors = NearestNeighbors(n_neighbors=20)
neighbors_fit = neighbors.fit(dataset)
distances, indices = neighbors_fit.kneighbors(dataset)


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()