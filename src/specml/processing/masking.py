# specml/processing/masking.py
import numpy as np

def create_mask_from_area(area: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """
    Creates a binary mask based on the area for each pixel's spectrum.

    Args:
        area (np.ndarray): Array of area values for each spectrum.
        threshold (float): Threshold value for creating the mask.

    Returns:
        np.ndarray: Binary mask where True indicates pixels with areas greater than the threshold.
    """
    return area > threshold


def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a binary mask to the dataset, retaining only the spectra for pixels marked as True.

    Args:
        data (np.ndarray): Dataset where each spectrum corresponds to a pixel.
        mask (np.ndarray): Binary mask indicating which spectra to retain.

    Returns:
        np.ndarray: Masked dataset containing only the spectra marked as True.
    """
    return np.array(data[mask])


def rebuild_tissue_mask(data, new_mask):
    """
    Rebuilds the tissue mask by mapping the new_mask (which is created from a subset of pixels)
    back to the full dataset mask (data.tissue_mask).
    
    :param data: SpectroscopicData object containing the original tissue mask.
    :param new_mask: The mask (binary array) created from a subset of pixels (e.g., tissue_mask_am1).
    :returns: Rebuilt mask with the new_mask applied to the original tissue mask.
    """
    # Initialize a new mask of ones with the same shape as the original tissue data

    # Flatten the 2D new_mask and data.mask if necessary
    if new_mask.ndim == 2:
        new_mask = new_mask.flatten()

    if data.mask and data.mask.ndim == 2:
        data.mask = data.mask.flatten()
    else:
        data.mask = np.ones((data.ypixels * data.xpixels,), dtype=bool)

    rebuild_mask = np.zeros((data.ypixels * data.xpixels,), dtype=bool).flatten()


    count = 0
    for i in np.arange(0, rebuild_mask.shape[0]):
        # Only replace values where the original tissue mask is 1 (active pixel)
        if data.mask[i] == 1:
            rebuild_mask[i] = new_mask[count]
            count += 1
    
    # Update the tissue mask in the data object with the rebuilt mask
    data.mask = rebuild_mask

    return rebuild_mask  # Return the rebuilt mask for further use, in its original format