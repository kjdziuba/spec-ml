import numpy as np

def arg_size_checker(upper, lower):
    """
    Ensures the upper and lower bounds are valid.
    If the upper is smaller than the lower, the values are swapped.

    :param upper: Upper bound
    :param lower: Lower bound
    :return: Corrected upper and lower bounds
    """
    if upper < lower:
        upper, lower = lower, upper  # Swap values if upper is less than lower
    return upper, lower

def get_wavenumber_indices(wavenumbers, lower, upper):
    """
    Returns the indices of the closest wavenumbers within the given limits.
    
    :param wavenumbers: Array of wavenumbers.
    :param lower: Lower wavenumber limit.
    :param upper: Upper wavenumber limit.
    :returns: Tuple (lowerlim, upperlim) of indices for the given wavenumber range.
    """
    upper, lower = arg_size_checker(upper, lower)

    if lower < wavenumbers[0] or upper > wavenumbers[-1]:
        print(f"Warning: Requested range ({lower}, {upper}) exceeds available wavenumbers.")
        lower = max(lower, wavenumbers[0])
        upper = min(upper, wavenumbers[-1])

    lowerlim = np.where(np.ravel(wavenumbers) >= lower)[0][0]  
    upperlim = np.where(np.ravel(wavenumbers) <= upper)[0][-1]

    return lowerlim, upperlim


def log_metadata(metadata_dict, key, message):
    """
    Logs metadata for transformations and operations performed on the data.

    :param metadata_dict: The metadata dictionary of the object.
    :param key: The operation or transformation key (e.g., 'PCA', 'UMAP').
    :param message: The description or detail about the transformation.
    """
    if 'logs' not in metadata_dict:
        metadata_dict['logs'] = []
    
    metadata_dict['logs'].append({key: message})


def flatten_3d_to_2d(data, mask=None):
    ypixels, xpixels, n_wavenumbers = data.shape
    flattened_data = data.reshape(-1, n_wavenumbers)
    if mask is not None:
        flattened_data = flattened_data[mask.flatten()]
    return flattened_data, (ypixels, xpixels)

def reshape_2d_to_3d(flattened_data, original_shape, mask=None):
    ypixels, xpixels = original_shape
    n_wavenumbers = flattened_data.shape[1]
    image_data = np.zeros((ypixels * xpixels, n_wavenumbers))
    if mask is not None:
        image_data[mask.flatten()] = flattened_data
    else:
        image_data = flattened_data
    image_data = image_data.reshape(ypixels, xpixels, n_wavenumbers)
    return image_data
