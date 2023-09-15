import faiss
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import cdist
# from sklearn.cluster import KMeans
from kmeans import Kmeans

class SpectralClustering:
    def __init__(self, n_clusters=8, n_neighbors=10):
        """
        Spectral clustering.

        Args:
        n_clusters (int): The number of clusters.
        n_neighbors (int): The number of nearest neighbors to consider for the graph construction.
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors

    def cluster(self, data):
        affinity_matrix = self.construct_affinity_matrix(data)
        laplacian_matrix = laplacian(affinity_matrix, normed=True)
        eigenvectors = self.calculate_eigenvectors(laplacian_matrix, self.n_clusters)
        kmeans = Kmeans(n_clusters=self.n_clusters, nredo=30)
        labels = kmeans.cluster(eigenvectors)

        return labels

    def construct_affinity_matrix(self, data):
        index = self.build_faiss_index(data)
        distances, neighbors = self.search_knn(index, data, self.n_neighbors)
        affinity_matrix = self.compute_gaussian_affinities(distances, neighbors)

        return affinity_matrix

    def build_faiss_index(self, data):
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, data.shape[1], flat_config)
        index.add(data.astype(np.float32))

        return index

    def search_knn(self, index, data, k):
        distances, neighbors = index.search(data.astype(np.float32), k + 1)

        return distances[:, 1:], neighbors[:, 1:]

    def compute_gaussian_affinities(self, distances, neighbors, sigma=1.0):
        n = distances.shape[0]
        affinity_matrix = np.zeros((n, n))
        gaussian_weights = np.exp(-np.square(distances) / (2 * (sigma ** 2)))

        for i in range(n):
            affinity_matrix[i, neighbors[i]] = gaussian_weights[i]

        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)

        return affinity_matrix

    def calculate_eigenvectors(self, laplacian_matrix, k):
        _, eigenvectors = np.linalg.eigh(laplacian_matrix)
        return eigenvectors[:, :k]
