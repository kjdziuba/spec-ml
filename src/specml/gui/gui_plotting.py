import numpy as np
import matplotlib.pyplot as plt

def plot_image_over_wavenumber_gui(spec_data_obj, target_wavenumber=None, wavenumber_idx=None, ax=None):
    """
    Visualizes the image from the SpectroscopicData object over a specified wavenumber or index.
    
    :param spec_data_obj: SpectroscopicData object containing data and wavenumbers
    :param target_wavenumber: Optional, the target wavenumber to visualize
    :param wavenumber_idx: Optional, the index of the wavenumber to visualize
    :param ax: The matplotlib axis where the plot should be drawn (for use in GUIs)
    """

    # Reshape data if necessary
    if len(spec_data_obj.data.shape) == 2:
        print("Data is flattened. Reshaping it back to 3D.")
        data = spec_data_obj.to_image()
    else:
        data = spec_data_obj.data

    wavenumbers = spec_data_obj.wavenumbers

    # Determine the image to plot
    if wavenumber_idx is not None:
        image = data[:, :, wavenumber_idx]
        ax.set_title(f'Image at Wavenumber Index: {wavenumber_idx}')
    elif target_wavenumber is not None:
        closest_idx = np.abs(wavenumbers - target_wavenumber).argmin()
        image = data[:, :, closest_idx]
        ax.set_title(f'Image at Wavenumber: {wavenumbers[closest_idx]} (Closest to {target_wavenumber})')
    else:
        raise ValueError("Either target_wavenumber or wavenumber_idx must be provided.")
    
    # Plot the image on the provided axis
    ax.imshow(image, cmap='viridis')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    print(f"Image plotted for wavenumber: {target_wavenumber} or index: {wavenumber_idx}")



def plot_dim_reduction_result(dim_result, labels=None, ax=None, cmap='viridis'):
    """
    Plots the results of a 2D dimensionality reduction (e.g., PCA, UMAP) with optional clustering labels.

    :param dim_result: DimReductionResult object containing reduced data.
    :param labels: Optional, cluster labels for coloring the points.
    :param ax: Matplotlib Axes object to plot on.
    :param cmap: Colormap to use for the plot (default is 'viridis').
    """
    if ax is None:
        ax = plt.gca()

    # Extract the reduced data from the DimReductionResult object
    reduction_result = dim_result.reduced_data
    
    # Check if labels are provided for coloring the scatter plot
    if labels is None:
        scatter = ax.scatter(reduction_result[:, 0], reduction_result[:, 1], c='blue', alpha=0.6, edgecolors='w')
    else:
        scatter = ax.scatter(reduction_result[:, 0], reduction_result[:, 1], c=labels, cmap=cmap, alpha=0.6, edgecolors='w')
        plt.colorbar(scatter, ax=ax, label="Cluster Labels")

    # Set the title and axis labels
    ax.set_title("Dimensionality Reduction Plot")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    plt.draw()  # Redraw the plot
