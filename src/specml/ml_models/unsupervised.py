import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def apply_kmeans(spec_data, n_clusters=5, scale_data=False, mask=None, original_shape=None):
    """
    Applies K-Means clustering to the data and returns a KMeansResult object.
    
    :param data: The dataset to cluster.
    :param n_clusters: Number of clusters to create. Default is 3.
    :param scale_data: Whether to scale data before clustering. Default is True.
    :param mask: Mask applied to the data.
    :param original_shape: Original (ypixels, xpixels) shape of the data.
    :returns: KMeansResult object.
    """
    data = spec_data.data
    if scale_data:
        data = MinMaxScaler().fit_transform(data)

    # Handle 3D data by flattening if necessary
    if data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
        
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(data)

    # Return the KMeansResult object with centroids, mask, and original shape
    return KMeansResult(cluster_labels, kmeans.cluster_centers_, mask, original_shape)



class KMeansResult:
    """
    A wrapper class to store KMeans clustering results along with metadata for reshaping and masking.
    """
    def __init__(self, cluster_labels, centroids, mask, original_shape):
        """
        Initialize with the cluster labels, centroids, mask, and the original (y, x) shape.
        
        :param cluster_labels: The KMeans cluster labels.
        :param centroids: The centroids of the clusters.
        :param mask: The mask applied to the original data (flattened mask).
        :param original_shape: The original shape of the data (ypixels, xpixels).
        """
        self.cluster_labels = cluster_labels
        self.centroids = centroids
        self.mask = mask
        self.original_shape = original_shape

    def to_image(self):
        """
        Reconstructs the cluster labels back into the original (y, x) shape.
        
        :returns: 2D array with the same spatial shape as the original image.
        """
        ypixels, xpixels = self.original_shape

        # Create an empty array with the original image shape
        image_data = np.zeros((ypixels * xpixels,))  # Flat image
        image_data[self.mask] = self.cluster_labels  # Fill only unmasked pixels

        # Reshape the flat image back to (y, x) shape
        return image_data.reshape((ypixels, xpixels))

    def get_centroids(self):
        """
        Returns the centroids of the KMeans clusters.
        """
        return self.centroids

    def get_cluster_labels(self):
        """
        Returns the flattened cluster labels.
        """
        return self.cluster_labels
