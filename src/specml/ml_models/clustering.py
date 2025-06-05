# specml/ml_models/clustering.py

from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

class BaseClustering(ABC):
    @abstractmethod
    def fit_predict(self, data):
        pass

    @abstractmethod
    def get_cluster_labels(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

class ClusteringResult:
    """
    A wrapper class to store clustering results along with metadata for reshaping and masking.
    """
    def __init__(self, cluster_labels, model, mask, original_shape):
        """
        Initialize with the cluster labels, model, mask, and the original (y, x) shape.

        :param cluster_labels: The clustering algorithm's labels.
        :param model: The fitted clustering model.
        :param mask: Mask applied to the data.
        :param original_shape: The original shape of the data (ypixels, xpixels).
        """
        self.cluster_labels = cluster_labels
        self.model = model
        self.mask = mask
        self.original_shape = original_shape

    def to_image(self):
        """
        Reconstructs the cluster labels back into the original (y, x) shape.

        :returns: 2D array with the same spatial shape as the original image.
        """
        ypixels, xpixels = self.original_shape

        # Create an empty array with the original image shape
        image_data = np.full((ypixels * xpixels,), fill_value=-1)  # Initialize with -1 for unassigned/masked pixels
        if self.mask is not None:
            image_data[self.mask] = self.cluster_labels
        else:
            image_data = self.cluster_labels

        # Reshape the flat image back to (y, x) shape
        return image_data.reshape((ypixels, xpixels))

# FUNCTION FOR REUSLTS USING MultiFileSpectroscopicData 
    def to_images(self):
        """
        Reconstructs the cluster labels back into the original shapes for each file.

        Returns:
            images (list): List of 2D arrays with cluster labels for each file.
        """
        images = []
        start_idx = 0
        for shape in self.original_shape:
            ypixels, xpixels = shape
            num_pixels = ypixels * xpixels
            end_idx = start_idx + num_pixels

            # Extract labels for this file
            labels = self.cluster_labels[start_idx:end_idx]

            # Handle masking
            if self.mask is not None:
                file_mask = self.mask[start_idx:end_idx]
                image_data = np.full((num_pixels,), fill_value=-1)
                image_data[file_mask] = labels
            else:
                image_data = labels

            # Reshape to original image shape
            image = image_data.reshape((ypixels, xpixels))
            images.append(image)

            start_idx = end_idx

        return images

    def get_cluster_labels(self):
        """
        Returns the flattened cluster labels.
        """
        return self.cluster_labels

    def get_model(self):
        """
        Returns the fitted clustering model.
        """
        return self.model

def apply_clustering(spec_data_obj, clustering_algorithm, mask=None):
    """
    Applies the specified clustering algorithm to the spectroscopic data.

    :param spec_data_obj: SpectroscopicData object containing data to cluster.
    :param clustering_algorithm: An instance of a clustering algorithm (e.g., KMeansClustering).
    :param mask: Mask applied to the data.
    :returns: ClusteringResult object.
    """
    data = spec_data_obj.data

    # Handle 3D data by flattening if necessary
    if data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
        # Apply mask if data was 3d and mask wasprovided
        if mask is not None:
            data = data[mask]

    # Fit the clustering algorithm
    cluster_labels = clustering_algorithm.fit_predict(data)

    # Create ClusteringResult
    clustering_result = ClusteringResult(
        cluster_labels=cluster_labels,
        model=clustering_algorithm.get_model(),
        mask=mask,
        original_shape=(spec_data_obj.ypixels, spec_data_obj.xpixels)
    )

    return clustering_result

class KMeansClustering(BaseClustering):
    def __init__(self, n_clusters=5, scale_data=False):
        """
        Initializes the KMeans clustering algorithm.

        :param n_clusters: Number of clusters to form.
        :param scale_data: Whether to scale data before clustering. Default is False.
        """
        self.n_clusters = n_clusters
        self.scale_data = scale_data
        self.model = None
        self.cluster_labels = None

    def fit_predict(self, data):
        """
        Fits the KMeans model to the data and predicts cluster labels.

        :param data: The dataset to cluster.
        :returns: Cluster labels.
        """
        if self.scale_data:
            data = MinMaxScaler().fit_transform(data)

        self.model = KMeans(n_clusters=self.n_clusters)
        self.cluster_labels = self.model.fit_predict(data)
        return self.cluster_labels

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_model(self):
        return self.model

class SpectralClusteringAlgorithm(BaseClustering):
    def __init__(self, n_clusters=5, scale_data=False):
        """
        Initializes the Spectral Clustering algorithm.

        :param n_clusters: Number of clusters to find.
        :param scale_data: Whether to scale data before clustering. Default is False.
        """
        self.n_clusters = n_clusters
        self.scale_data = scale_data
        self.model = None
        self.cluster_labels = None

    def fit_predict(self, data):
        """
        Fits the Spectral Clustering model to the data and predicts cluster labels.

        :param data: The dataset to cluster.
        :returns: Cluster labels.
        """
        if self.scale_data:
            data = MinMaxScaler().fit_transform(data)

        self.model = SpectralClustering(n_clusters=self.n_clusters, affinity='nearest_neighbors', assign_labels='kmeans')
        self.cluster_labels = self.model.fit_predict(data)
        return self.cluster_labels

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_model(self):
        return self.model
    


### Memory efficient models
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch

class MiniBatchKMeansClustering(BaseClustering):
    def __init__(self, n_clusters=5, batch_size=1000, scale_data=False):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.scale_data = scale_data
        self.model = None
        self.cluster_labels = None

    def fit_predict(self, data):
        if self.scale_data:
            data = MinMaxScaler().fit_transform(data)
        self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size)
        self.cluster_labels = self.model.fit_predict(data)
        return self.cluster_labels

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_model(self):
        return self.model

class BirchClustering(BaseClustering):
    def __init__(self, n_clusters=5, threshold=0.5, scale_data=False):
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.scale_data = scale_data
        self.model = None
        self.cluster_labels = None

    def fit_predict(self, data):
        if self.scale_data:
            data = MinMaxScaler().fit_transform(data)
        self.model = Birch(n_clusters=self.n_clusters, threshold=self.threshold)
        self.cluster_labels = self.model.fit_predict(data)
        return self.cluster_labels

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_model(self):
        return self.model
