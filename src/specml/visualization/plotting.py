# src/specml/visualization/plotting.py
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

def display_image(image_data, ypixels, xpixels, mask=None, title=None, color_bar=False, greyscale=False, cmap="viridis"):
    """
    Displays a 2D image from the given data, ensuring it fits the original (ypixels, xpixels) shape.
    
    :param image_data: Flattened or reshaped image data.
    :param ypixels: Number of y-pixels in the original image.
    :param xpixels: Number of x-pixels in the original image.
    :param mask: Optional, mask used to display only unmasked pixels.
    :param title: Optional, title of the plot.
    :param color_bar: Optional, if True, adds a color bar.
    :param greyscale: Optional, if True, displays in greyscale.
    :param cmap: Colormap to use.
    """
    if image_data.ndim == 1:
        # If the mask is provided, rebuild the image with the mask
        if mask is not None:
            full_image = np.zeros(ypixels * xpixels)  # Flat array for full image
            full_image[mask] = image_data  # Place unmasked pixels
            image_data = full_image.reshape((ypixels, xpixels))  # Reshape to original dimensions
        else:
            # Reshape without a mask
            image_data = image_data.reshape((ypixels, xpixels))

    # Set figure size and plot
    fig, ax = plt.subplots(figsize=(15.36, 5.12))
    cmap_used = 'gray' if greyscale else cmap
    im = ax.imshow(image_data, cmap=cmap_used)

    # Add color bar if requested
    if color_bar:
        plt.colorbar(im, ax=ax)

    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Remove axis for a cleaner view
    ax.axis('off')

    plt.show()
    return im.figure

#used for multiple iamges for multi_spectroscopic_data
def display_images(image_data_list, ypixels_list, xpixels_list, masks=None, titles=None, color_bar=False, greyscale=False, cmap="viridis"):
    """
    Displays multiple 2D images from the given data.

    :param image_data_list: List of image data arrays.
    :param ypixels_list: List of ypixels for each image.
    :param xpixels_list: List of xpixels for each image.
    :param masks: Optional list of masks for each image.
    :param titles: Optional list of titles for each image.
    :param color_bar: Whether to include a color bar.
    :param greyscale: Whether to use a greyscale colormap.
    :param cmap: Colormap to use.
    """
    num_images = len(image_data_list)
    fig, axes = plt.subplots(1, num_images, figsize=(15 * num_images, 5))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, ax in enumerate(axes):
        image_data = image_data_list[idx]
        ypixels = ypixels_list[idx]
        xpixels = xpixels_list[idx]
        mask = masks[idx] if masks else None
        title = titles[idx] if titles else None

        if image_data.ndim == 1:
            if mask is not None:
                full_image = np.zeros(ypixels * xpixels)
                full_image[mask] = image_data
                image_data = full_image.reshape((ypixels, xpixels))
            else:
                image_data = image_data.reshape((ypixels, xpixels))

        cmap_used = 'gray' if greyscale else cmap
        im = ax.imshow(image_data, cmap=cmap_used)
        if color_bar:
            plt.colorbar(im, ax=ax)
        if title:
            ax.set_title(title)
        ax.axis('off')

    plt.show()


# General Exploratory Plotting functions
def plot_image_over_wavenumber(spec_data_obj, target_wavenumber=None, wavenumber_idx=None):
    """
    Visualizes the image from the SpectroscopicData object over a specified wavenumber or index.
    
    :param spec_data_obj: SpectroscopicData object containing data and wavenumbers
    :param target_wavenumber: Optional, the target wavenumber to visualize
    :param wavenumber_idx: Optional, the index of the wavenumber to visualize
    """

    # If data is flattened, reshape it back to 3D
    if len(spec_data_obj.data.shape) == 2:
        print("Data is flattened. Reshaping it back to 3D.")
        data = spec_data_obj.to_image()
    else:
        # Fetch the data and wavenumbers from the object
        data = spec_data_obj.data
        
    wavenumbers = spec_data_obj.wavenumbers
    
    if wavenumber_idx is not None:
        # If an index is provided, use it
        image = data[:, :, wavenumber_idx]
        plt.title(f'Image at Wavenumber Index: {wavenumber_idx}')
    elif target_wavenumber is not None:
        # Find the closest wavenumber if a target wavenumber is provided
        closest_idx = np.abs(wavenumbers - target_wavenumber).argmin()
        image = data[:, :, closest_idx]
        plt.title(f'Image at Wavenumber: {wavenumbers[closest_idx]:.2f} (Closest to {target_wavenumber})')
    else:
        raise ValueError("Either target_wavenumber or wavenumber_idx must be provided.")
    
    print(f"Image slice shape: {image.shape}")

    # Plot the image
    plt.imshow(image, cmap='viridis')
    # plt.colorbar()
    plt.show()

def plot_mean_spectrum_with_confidence_interval(
    spec_data_obj,
    confidence_level=0.90,
    labels=None,
    shade=True,
    title=None,
    **kwargs
):
    """
    Plots the mean spectrum(s) with shaded confidence intervals.
    Assumes normal distributions and calculates confidence intervals.

    Args:
        spec_data_obj (SpectroscopicData): The spectroscopic data object.
        confidence_level (float): Confidence level for the interval (between 0 and 1). Default is 0.90.
        labels (array-like, optional): Labels for grouping spectra. Should be a 1D array with group identifiers.
        shade (bool): Whether to shade the confidence interval. Default is True.
        title (str, optional): Title for the plot.
        **kwargs: Additional keyword arguments for plotting customization.
    """
    # Flatten data and apply mask
    data = spec_data_obj.flatten()
    wavenumbers = spec_data_obj.wavenumbers

    if labels is not None:
        labels = np.array(labels)
        if labels.shape[0] != data.shape[0]:
            raise ValueError("Labels array must have the same length as the number of spectra.")
        unique_labels = np.unique(labels)
    else:
        unique_labels = [None]  # Single group

    # Calculate the z-score for the desired confidence level
    z_score = stats.norm.ppf(0.5 + confidence_level / 2)

    # Initialize plot
    plt.figure(figsize=kwargs.get('figsize', (10, 6)))

    for label in unique_labels:
        if label is not None:
            # Select data for the current group
            group_data = data[labels == label]
            label_name = f'Group {label}'
        else:
            # All data if no labels provided
            group_data = data
            label_name = 'Mean Spectrum'

        if group_data.shape[0] < 2:
            raise ValueError("At least two spectra are required to calculate confidence intervals.")

        # Calculate mean and standard deviation
        mean_spectrum = np.mean(group_data, axis=0)
        std_spectrum = np.std(group_data, axis=0, ddof=1)

        # Calculate confidence interval using standard deviation
        h = std_spectrum * z_score

        # Plot mean spectrum
        plt.plot(wavenumbers, mean_spectrum, label=label_name)

        # Plot confidence interval
        if shade:
            plt.fill_between(
                wavenumbers,
                mean_spectrum - h,
                mean_spectrum + h,
                alpha=0.3,
                label=f'{int(confidence_level * 100)}% Confidence Interval' if label is None else None
            )

    plt.xlabel(kwargs.get('xlabel', 'Wavenumber (cm⁻¹)'))
    plt.ylabel(kwargs.get('ylabel', 'Absorbance'))
    if title:
        plt.title(title)
    else:
        plt.title('Mean Spectrum with Confidence Interval')
    plt.legend()
    plt.gca().invert_xaxis()  # Invert x-axis if wavenumbers are in descending order
    plt.show()

def plot_value_distribution(values, bins='auto', title='Value Distribution', xlabel='Value', ylabel='Frequency'):
    """
    Plots the distribution of an array of values using a histogram and KDE.

    Args:
        values (array-like): Array of values to plot.
        bins (int): Number of bins for the histogram. Default is 30.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    print(f"Values: {values.shape}")
    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=bins, kde=True, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_area_map(spec_data_obj, lower_wavenumber, upper_wavenumber, cmap='viridis'):
    """
    Plots a map of the integrated area under the spectra between specified wavenumbers.

    Args:
        spec_data_obj (SpectroscopicData): The spectroscopic data object.
        lower_wavenumber (float): Lower bound of the wavenumber range.
        upper_wavenumber (float): Upper bound of the wavenumber range.
        cmap (str): Colormap for the heatmap. Default is 'viridis'.
    """
    # Calculate areas
    areas = spec_data_obj.calculate_area(lower=lower_wavenumber, upper=upper_wavenumber)
    # Reshape areas to image
    area_image = np.zeros((spec_data_obj.ypixels * spec_data_obj.xpixels))
    area_image[spec_data_obj.mask] = areas
    area_image = area_image.reshape(spec_data_obj.ypixels, spec_data_obj.xpixels, -1)
    plt.figure(figsize=(8, 6))
    plt.imshow(area_image, cmap=cmap)
    plt.title(f'Integrated Area Map ({lower_wavenumber}-{upper_wavenumber} cm⁻¹)')
    plt.colorbar(label='Area')
    plt.axis('off')
    plt.show()

def plot_individual_spectra(spec_data_obj, coordinates=None, num_spectra=5):
    """
    Plots individual spectra from the dataset.

    Args:
        spec_data_obj (SpectroscopicData): The spectroscopic data object.
        coordinates (list of tuples, optional): List of (x, y) coordinates of spectra to plot.
        num_spectra (int): Number of random spectra to plot if coordinates are not provided. Default is 5.
    """
    wavenumbers = spec_data_obj.wavenumbers

    if coordinates is not None:
        spectra = []
        for x, y in coordinates:
            spectrum = spec_data_obj.get_spectrum_at(x, y)
            spectra.append((spectrum, f'Spectrum at ({x}, {y})'))
    else:
        # Randomly select spectra if coordinates are not provided
        data = spec_data_obj.flatten()
        indices = np.random.choice(data.shape[0], num_spectra, replace=False)
        spectra = [(data[idx], f'Spectrum {idx}') for idx in indices]

    plt.figure(figsize=(10, 6))
    for spectrum, label in spectra:
        plt.plot(wavenumbers, spectrum, label=label)

    plt.xlabel('Wavenumber')
    plt.ylabel('Absorbance')
    plt.title('Individual Spectra')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()

def plot_waterfall(spec_data_obj, indices=None, title='Waterfall Plot', cmap='viridis'):
    """
    Creates a 3D waterfall plot of multiple spectra.

    Args:
        spec_data_obj (SpectroscopicData): The spectroscopic data object.
        indices (list of int, optional): Indices of spectra to plot. If None, plots all spectra.
        title (str): Title of the plot.
        cmap (str): Colormap for the plot.
    """
    data = spec_data_obj.flatten()
    wavenumbers = spec_data_obj.wavenumbers

    if indices is None:
        indices = range(200)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(indices)))

    for i, idx in enumerate(indices):
        spectrum = data[idx]
        xs = wavenumbers
        ys = spectrum
        zs = i  # Use the index as the z-coordinate
        ax.plot(xs, zs, ys, color=colors[i])

    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Spectrum Index')
    ax.set_zlabel('Absorbance')
    ax.set_title(title)
    ax.invert_xaxis()
    plt.show()


# PCA and UMAP plotting
def plot_dim_reduction_result(dim_result, labels=None, label_mapping=None, title="Dimensionality Reduction Result", cmap='viridis', size=None, alpha=0.3):
    """
    Plots the results of a 2D dimensionality reduction (e.g., PCA, UMAP) with optional clustering labels.
    
    :param dim_result: DimReductionResult object containing reduced data.
    :param labels: Optional, labels for coloring the points.
    :param title: The title of the plot.
    :param cmap: Colormap for plotting.
    """
    reduction_result = dim_result.reduced_data
    labels = dim_result.labels
    plt.figure(figsize=(8, 6))
    
    # If no labels are provided, plot all points in the same color
    if labels is None:
        plt.scatter(reduction_result[:, 0], reduction_result[:, 1], c='blue', alpha=alpha, edgecolors='w', s=size)
    else:
        scatter = plt.scatter(reduction_result[:, 0], reduction_result[:, 1], c=labels, cmap=cmap, alpha=alpha, edgecolors='w', s=size)
        if label_mapping:
            cbar = plt.colorbar(scatter, ticks=np.unique(labels))
            cbar.set_label("Class Labels")
            cbar.set_ticklabels([label_mapping[label] for label in np.unique(labels)])
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

def plot_scree(dim_result, title="Scree Plot"):
    """
    Plots a scree plot showing the explained variance ratio of the PCA components.
    
    :param pca: The fitted PCA object.
    :param title: Title of the plot.
    """

    pca = dim_result.model
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure()
    plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.title(title)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

def plot_3d_pca(dim_result, labels=None, label_mapping=None, title="3D PCA Plot", cmap='viridis', size=None, alpha=0.3):
    """
    Plots 3D PCA results from DimReductionResult.
    
    :param dim_result: DimReductionResult object containing PCA transformed data.
    :param labels: Optional, cluster labels for coloring the points.
    :param title: Title of the plot.
    :param cmap: Color map for plotting.
    """
    pca_data = dim_result.reduced_data      
    labels = dim_result.labels
    plt.figure(figsize=(8, 6))

    if labels is None:
        plt.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c='blue', alpha=alpha, edgecolors='w', s=size)
        
    else:
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=labels, cmap=cmap, alpha=alpha, edgecolors='w', s=size)
        if label_mapping:
            cbar = plt.colorbar(scatter, ticks=np.unique(labels))
            cbar.set_label("Class Labels")
            cbar.set_ticklabels([label_mapping[label] for label in np.unique(labels)])
        
    plt.set_title(title)
    plt.set_xlabel("Component 1")
    plt.set_ylabel("Component 2")
    plt.set_zlabel("Component 3")
    plt.show()

def plot_correlation_heatmap(data, title="Correlation Heatmap", wavenumbers=None, annot=False):
    """
    Plots a heatmap of the feature correlations in the dataset.
    
    :param data: The dataset for which correlations are plotted.
    :param title: The title of the plot.
    :param wavenumbers: Optional. The wavenumbers to display on the axes.
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = np.corrcoef(data.T)  # Correlation matrix
    
    if wavenumbers is not None:
        sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', fmt='.2f', 
                    linewidths=0.5, xticklabels=wavenumbers, yticklabels=wavenumbers)
    else:
        sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    
    plt.title(title)
    plt.xlabel("Wavenumbers" if wavenumbers is not None else "Features")
    plt.ylabel("Wavenumbers" if wavenumbers is not None else "Features")
    plt.show()

def plot_variance_explained(dim_result, title="Variance Explained by PCA Components"):
    """
    Plots the variance explained by each PCA component.
    
    :param pca: The fitted PCA object.
    :param title: The title of the plot.
    """

    pca = dim_result.model
    plt.figure()
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.title(title)
    plt.show()

def plot_pca_loadings(dim_result, component=1, title=None):
    """
    Plots the loadings of the specified principal component.

    Args:
        dim_result (DimReductionResult): Object containing PCA model and wavenumbers.
        component (int): Principal component number to plot (1-indexed).
        title (str, optional): Title of the plot.

    Raises:
        ValueError: If the specified component index is out of bounds.
    """
    pca = dim_result.model
    wavenumbers = dim_result.wavenumbers

    if component < 1 or component > pca.components_.shape[0]:
        raise ValueError(f"Component index out of range. Valid range is 1 to {pca.components_.shape[0]}.")

    loadings = pca.components_[component - 1]

    plt.figure(figsize=(10, 6))
    plt.plot(wavenumbers, loadings)
    plt.xlabel('Wavenumber')
    plt.ylabel('Loading')
    plt.title(title or f'PCA Component {component} Loadings')
    plt.gca().invert_xaxis()
    plt.show()

def display_dim_reduction_as_image(dim_result, component_idx=0, title="Dimensionality Reduction", cmap="viridis"):
    """
    Displays dimensionality reduction results (e.g., PCA, UMAP) as an image over the original (y, x) layout.
    
    :param dim_result: DimReductionResult object containing reduced data.
    :param component_idx: Index of the component to display (e.g., for PCA or UMAP results).
    :param title: Title of the plot.
    :param cmap: Colormap to use.
    """
    # Use the first principal component or specified component
    component_image = dim_result.to_image(component_idx)

    # Get original shape from DimReductionResult
    ypixels, xpixels = dim_result.original_shape

    # Display the reshaped component image in the original image layout
    return display_image(component_image, ypixels, xpixels, title=title, color_bar=True, cmap=cmap)

def plot_pca_residuals(dim_result, original_data, title='PCA Residuals Plot'):
    """
    Plots Q-residuals vs Hotelling's T-squared statistics.

    Args:
        dim_result (DimReductionResult): Object containing PCA model, reduced data, and original data.
        title (str): Title of the plot.
    """
    pca = dim_result.model
    scores = dim_result.reduced_data
    reconstructed = pca.inverse_transform(scores)
    residuals = original_data.data - reconstructed
    q_residuals = np.sum(residuals ** 2, axis=1)
    t_squared = np.sum((scores ** 2) / pca.explained_variance_, axis=1)

    plt.figure(figsize=(8, 6))
    plt.scatter(t_squared, q_residuals, alpha=0.7, edgecolors='k')
    plt.xlabel("Hotelling's T-squared")
    plt.ylabel('Q-Residuals')
    plt.title(title)
    plt.grid(True)
    plt.show()

