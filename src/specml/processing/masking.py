# specml/processing/masking.py
import numpy as np
from skimage.morphology import binary_closing, disk, remove_small_objects

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
    """Project ``new_mask`` (defined over currently-active pixels) back onto
    the full image mask.

    ``new_mask`` has one entry per pixel that was active in ``data.mask``;
    the result is a 1-D mask over all ``ypixels * xpixels`` pixels where each
    entry is True only when the original mask was active AND the corresponding
    new_mask entry is True. ``data.mask`` is updated in place.

    Args:
        data: ``SpectroscopicData`` instance carrying the current tissue mask.
        new_mask: Boolean mask, length equal to the number of active pixels
            in ``data.mask`` (or to ``ypixels * xpixels`` when no mask is set).

    Returns:
        The rebuilt 1-D mask (also written to ``data.mask``).
    """
    new_mask = np.asarray(new_mask).ravel().astype(bool)

    total_pixels = int(data.ypixels) * int(data.xpixels)
    if data.mask is None:
        data.mask = np.ones(total_pixels, dtype=bool)
    else:
        data.mask = np.asarray(data.mask).ravel().astype(bool)
        if data.mask.size != total_pixels:
            raise ValueError(
                f"rebuild_tissue_mask: existing mask has {data.mask.size} entries "
                f"but ypixels*xpixels = {total_pixels}."
            )

    active_indices = np.flatnonzero(data.mask)
    if new_mask.size != active_indices.size:
        raise ValueError(
            f"rebuild_tissue_mask: new_mask has {new_mask.size} entries but "
            f"original mask has {active_indices.size} active pixels."
        )

    rebuild_mask = np.zeros(total_pixels, dtype=bool)
    rebuild_mask[active_indices] = new_mask
    data.mask = rebuild_mask
    return rebuild_mask


def clean_tissue_mask_spatial(
    mask,
    ypixels,
    xpixels,
    *,
    min_size_px=None,
    closing_radius_px=0,
):
    """Spatial connected-component clean-up of a flat tissue mask.

    The rat-lung preprocessing baseline (see
    ``rat_lung_project/configs/preprocessing_baseline.yaml``) asks for
    ``remove_small_objects`` thresholds of 400 px (FTIR) / 1200 px (QCL)
    after the Amide-I integration threshold builds the initial mask.
    Optionally also runs a morphological closing with a disk-shaped
    structuring element of radius ``closing_radius_px`` to bridge small
    gaps inside the tissue.

    Both inputs and outputs are 1-D flat masks of length
    ``ypixels * xpixels``. ``min_size_px`` set to ``None`` or ``0`` skips the
    size threshold; ``closing_radius_px=0`` skips the closing.
    """
    flat = np.asarray(mask).ravel().astype(bool)
    expected = int(ypixels) * int(xpixels)
    if flat.size != expected:
        raise ValueError(
            f"clean_tissue_mask_spatial: mask has {flat.size} entries but "
            f"ypixels*xpixels = {expected}."
        )

    mask_2d = flat.reshape(int(ypixels), int(xpixels))

    if closing_radius_px and closing_radius_px > 0:
        mask_2d = binary_closing(mask_2d, footprint=disk(int(closing_radius_px)))

    if min_size_px:
        # skimage 0.26 deprecated `min_size` in favour of `max_size`. The new
        # `max_size=N` removes components with size <= N (keeps size > N),
        # whereas the old `min_size=N` removed components with size < N
        # (kept size >= N). The YAML semantic is "keep components with at
        # least N pixels", so we pass max_size = min_size - 1.
        mask_2d = remove_small_objects(mask_2d, max_size=int(min_size_px) - 1)

    return mask_2d.astype(bool).ravel()