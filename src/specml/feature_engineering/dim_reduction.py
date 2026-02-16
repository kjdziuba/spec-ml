import numpy as np
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from specml.ml_models.clustering import ClusteringResult

def apply_pca(spectro_data, n_components=2, scale_data=False):
    """
    Applies PCA to a SpectroscopicData object and returns a DimReductionResult.
    
    :param spectro_data: SpectroscopicData object.
    :param n_components: Number of principal components to reduce to. Default is 2.
    :param scale_data: Whether to scale data before applying PCA. Default is True.
    :returns: DimReductionResult object.
    """

    data = MinMaxScaler().fit_transform(spectro_data.data) if scale_data else spectro_data.data

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    return PCAResult(pca_data, spectro_data.mask, spectro_data.wavenumbers, (spectro_data.ypixels, spectro_data.xpixels), model=pca, labels=spectro_data.labels)


def apply_umap(spectro_data, n_components=2, scale_data=False, random_state=None):
    """
    Applies UMAP to a SpectroscopicData object and returns a DimReductionResult.
    
    :param spectro_data: SpectroscopicData object.
    :param n_components: Number of dimensions to reduce to. Default is 2.
    :param scale_data: Whether to scale data before applying UMAP. Default is True.
    :param random_state: Set a random state for reproducibility. Default is None.
    :returns: DimReductionResult object.
    """
    # data = spectro_data.flatten()

    data = MinMaxScaler().fit_transform(spectro_data.data) if scale_data else spectro_data.data

    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    umap_data = reducer.fit_transform(data)
    return UMAPResult(umap_data, spectro_data.mask, spectro_data.wavenumbers, (spectro_data.ypixels, spectro_data.xpixels), model=reducer)


class DimReductionResult:
    """
    A wrapper class to store dimensionality reduction or clustering results along with metadata.
    """
    def __init__(self, reduced_data, mask, wavenumbers, original_shape, model=None, labels=None):
        self.reduced_data = reduced_data
        self.mask = mask
        self.original_shape = original_shape
        self.model = model
        self.labels = labels
        self.wavenumbers = wavenumbers

    def to_image(self, component_idx=0):
        """
        Reconstructs the specified component back into the original (y, x) shape.
        """
        ypixels, xpixels = self.original_shape
        image_data = np.zeros((ypixels * xpixels,), dtype=self.reduced_data.dtype)
        image_data[self.mask.flatten()] = self.reduced_data[:, component_idx]
        return image_data.reshape((ypixels, xpixels))


    def get_component(self, component_idx=0):
        """
        Returns the component data in its flattened form (without reshaping to image).
        
        :param component_idx: Index of the component to return.
        :returns: Flattened array of the component data.
        """
        return self.reduced_data[:, component_idx]
    
    def cluster_data(self, clustering_algorithm):
        """
        Applies the specified clustering algorithm to the reduced data.

        Args:
            clustering_algorithm: An instance of a clustering algorithm (e.g., KMeansClustering).

        Returns:
            ClusteringResult: The result of the clustering.
        """
        data = self.reduced_data


        # Fit the clustering algorithm
        cluster_labels = clustering_algorithm.fit_predict(data)

        # Create ClusteringResult
        clustering_result = ClusteringResult(
            cluster_labels=cluster_labels,
            model=clustering_algorithm.get_model(),
            mask=self.mask,
            original_shape= self.original_shape
        )

        return clustering_result

# PCA-specific class
class PCAResult(DimReductionResult):
    def explained_variance_ratio(self):
        return self.model.explained_variance_ratio_

# UMAP-specific class
class UMAPResult(DimReductionResult):
    def get_embedding(self):
        return self.model.embedding_