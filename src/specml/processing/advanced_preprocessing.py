"""
Advanced preprocessing functions for FTIR spectroscopic data.
Specialized for biological tissue analysis and lesion detection.
"""

import numpy as np
import logging
from scipy import signal, ndimage
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)

def snv_normalization(data):
    """
    Standard Normal Variate (SNV) normalization.
    
    Removes multiplicative interferences caused by scattering effects
    and path length variations. Essential for biological tissue analysis
    where sample thickness varies.
    
    Chemistry: Corrects for physical sample variations while preserving
    chemical information. Particularly useful for tissue samples where
    thickness and surface scattering effects are significant.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
    
    Returns:
        np.ndarray: SNV-normalized data
    """
    # Calculate mean and std for each spectrum
    mean_spectrum = np.mean(data, axis=1, keepdims=True)
    std_spectrum = np.std(data, axis=1, keepdims=True)
    
    # Avoid division by zero
    std_spectrum[std_spectrum == 0] = 1
    
    # Apply SNV normalization
    snv_data = (data - mean_spectrum) / std_spectrum
    
    logger.info("Applied SNV normalization")
    return snv_data

def msc_normalization(data, reference_spectrum=None):
    """
    Multiplicative Scatter Correction (MSC) normalization.
    
    Corrects for multiplicative scattering effects using a reference spectrum.
    Particularly effective for biological samples where scattering varies
    significantly between measurements.
    
    Chemistry: Removes physical light scattering effects while preserving
    chemical absorption features. Uses linear regression to correct each
    spectrum relative to a reference.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        reference_spectrum (np.ndarray, optional): Reference spectrum for correction
    
    Returns:
        np.ndarray: MSC-corrected data
    """
    if reference_spectrum is None:
        reference_spectrum = np.mean(data, axis=0)
    
    corrected_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        # Linear regression: spectrum = a * reference + b
        coeffs = np.polyfit(reference_spectrum, data[i], 1)
        a, b = coeffs[0], coeffs[1]
        
        # Apply MSC correction
        corrected_data[i] = (data[i] - b) / a
    
    logger.info("Applied MSC normalization")
    return corrected_data

def extended_multiplicative_scatter_correction(data, reference_spectrum=None):
    """
    Extended Multiplicative Scatter Correction (EMSC).
    
    Advanced version of MSC that accounts for both scattering and
    chemical interference effects. Particularly useful for biological
    tissue analysis where water content varies.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        reference_spectrum (np.ndarray, optional): Reference spectrum
    
    Returns:
        np.ndarray: EMSC-corrected data
    """
    if reference_spectrum is None:
        reference_spectrum = np.mean(data, axis=0)
    
    corrected_data = np.zeros_like(data)
    wavenumbers = np.arange(data.shape[1])
    
    for i in range(data.shape[0]):
        # Extended model: spectrum = a * reference + b + c * wavenumber + d * wavenumber^2
        X = np.column_stack([
            reference_spectrum,
            np.ones(len(reference_spectrum)),
            wavenumbers,
            wavenumbers**2
        ])
        
        # Least squares fit
        coeffs = np.linalg.lstsq(X, data[i], rcond=None)[0]
        a, b, c, d = coeffs
        
        # Apply EMSC correction (keep only chemical information)
        corrected_data[i] = (data[i] - b - c * wavenumbers - d * wavenumbers**2) / a
    
    logger.info("Applied EMSC normalization")
    return corrected_data

def detrend_spectrum(data, method='linear'):
    """
    Removes linear or polynomial trends from spectra.
    
    Essential for removing baseline drift caused by instrumental
    drift or sample positioning changes during measurement.
    
    Chemistry: Removes systematic baseline drift while preserving
    peak positions and relative intensities.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        method (str): Detrending method ('linear', 'quadratic', 'cubic')
    
    Returns:
        np.ndarray: Detrended data
    """
    detrended_data = np.zeros_like(data)
    x = np.arange(data.shape[1])
    
    degree_map = {'linear': 1, 'quadratic': 2, 'cubic': 3}
    degree = degree_map.get(method, 1)
    
    for i in range(data.shape[0]):
        # Fit polynomial trend
        trend_coeffs = np.polyfit(x, data[i], degree)
        trend = np.polyval(trend_coeffs, x)
        
        # Remove trend
        detrended_data[i] = data[i] - trend
    
    logger.info(f"Applied {method} detrending")
    return detrended_data

def rubberband_baseline_correction(data, num_points=100):
    """
    Rubberband baseline correction using convex hull.
    
    Effective for biological samples where baseline curvature
    varies significantly. Particularly useful for tissue samples
    with varying thickness and composition.
    
    Chemistry: Estimates baseline by finding the convex hull of
    the spectrum, representing the minimum possible baseline.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        num_points (int): Number of points to use for convex hull
    
    Returns:
        np.ndarray: Baseline-corrected data
    """
    from scipy.spatial import ConvexHull
    
    corrected_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        spectrum = data[i]
        x = np.arange(len(spectrum))
        
        # Subsample for convex hull calculation
        indices = np.linspace(0, len(spectrum)-1, num_points, dtype=int)
        points = np.column_stack([x[indices], -spectrum[indices]])  # Negative for convex hull
        
        # Calculate convex hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        # Sort by x-coordinate
        hull_points = hull_points[hull_points[:, 0].argsort()]
        
        # Interpolate baseline
        baseline = np.interp(x, hull_points[:, 0], -hull_points[:, 1])
        
        # Subtract baseline
        corrected_data[i] = spectrum - baseline
    
    logger.info("Applied rubberband baseline correction")
    return corrected_data

def als_baseline_correction(data, lam=10**6, p=0.01, niter=10):
    """
    Asymmetric Least Squares baseline correction.
    
    Effective for complex baseline shapes in biological samples.
    Asymmetric weighting ensures the baseline follows the bottom
    of the spectrum while avoiding peaks.
    
    Chemistry: Distinguishes between positive peaks (absorption)
    and negative baseline drift, making it ideal for FTIR spectra
    of biological samples.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        lam (float): Smoothing parameter (higher = smoother baseline)
        p (float): Asymmetry parameter (0 < p < 1)
        niter (int): Number of iterations
    
    Returns:
        np.ndarray: Baseline-corrected data
    """
    corrected_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        spectrum = data[i]
        L = len(spectrum)
        D = np.diff(np.eye(L), 2)
        w = np.ones(L)
        
        for _ in range(niter):
            W = np.diag(w)
            Z = W + lam * D.T @ D
            z = np.linalg.solve(Z, w * spectrum)
            w = p * (spectrum > z) + (1 - p) * (spectrum < z)
        
        corrected_data[i] = spectrum - z
    
    logger.info("Applied ALS baseline correction")
    return corrected_data

def remove_water_vapor_interference(data, wavenumbers, regions=None):
    """
    Removes water vapor interference from FTIR spectra.
    
    Water vapor creates characteristic absorption bands that
    can interfere with biological tissue analysis. This function
    identifies and removes these interferences.
    
    Chemistry: Water vapor shows distinct rotational-vibrational
    bands around 3500-4000 cm⁻¹ and 1300-1900 cm⁻¹. These can
    mask important biological peaks.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        wavenumbers (np.ndarray): Wavenumber values
        regions (list): List of (start, end) tuples for water vapor regions
    
    Returns:
        np.ndarray: Data with water vapor interference removed
    """
    if regions is None:
        # Default water vapor regions
        regions = [(3500, 4000), (1300, 1900)]
    
    corrected_data = data.copy()
    
    for start, end in regions:
        # Find indices for water vapor region
        region_mask = (wavenumbers >= start) & (wavenumbers <= end)
        region_indices = np.where(region_mask)[0]
        
        if len(region_indices) > 0:
            # Interpolate across the region
            for i in range(data.shape[0]):
                if len(region_indices) > 2:
                    # Find boundary points
                    left_idx = max(0, region_indices[0] - 5)
                    right_idx = min(len(wavenumbers) - 1, region_indices[-1] + 5)
                    
                    # Linear interpolation
                    x_interp = [left_idx, right_idx]
                    y_interp = [data[i, left_idx], data[i, right_idx]]
                    
                    # Replace region with interpolated values
                    for j in region_indices:
                        corrected_data[i, j] = np.interp(j, x_interp, y_interp)
    
    logger.info(f"Removed water vapor interference in {len(regions)} regions")
    return corrected_data

def rmies_emsc_correction(data, wavenumbers, n_iterations=10, reference_spectrum=None):
    """
    RMieS-EMSC (Resonant Mie Scattering - Extended Multiplicative Scatter Correction).
    
    Advanced scattering correction that accounts for both Rayleigh and Mie scattering
    effects. Particularly important for biological tissues where particle sizes
    are comparable to IR wavelengths.
    
    Used by Gardner and Bassan groups for tissue analysis.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        wavenumbers (np.ndarray): Wavenumber values
        n_iterations (int): Number of iterative corrections (default: 10)
        reference_spectrum (np.ndarray, optional): Reference spectrum
    
    Returns:
        np.ndarray: RMieS-EMSC corrected data
    """
    if reference_spectrum is None:
        reference_spectrum = np.mean(data, axis=0)
    
    corrected_data = data.copy()
    
    for iteration in range(n_iterations):
        # Mie scattering model: extinction ~ 1/wavelength^q
        # For biological tissues, q typically ranges from 1-4
        wavelengths = 1e4 / wavenumbers  # Convert to micrometers
        
        for i in range(data.shape[0]):
            spectrum = corrected_data[i]
            
            # Extended model with Mie scattering terms
            X = np.column_stack([
                reference_spectrum,                    # Chemical information
                np.ones(len(reference_spectrum)),      # Offset
                wavelengths**(-1),                     # Rayleigh scattering
                wavelengths**(-2),                     # Mie scattering q=2
                wavelengths**(-3),                     # Mie scattering q=3
                wavelengths**(-4)                      # Mie scattering q=4
            ])
            
            # Robust least squares fit
            try:
                coeffs = np.linalg.lstsq(X, spectrum, rcond=None)[0]
                a, b, c1, c2, c3, c4 = coeffs
                
                # Remove scattering effects, keep chemical information
                scattering_component = (b + c1*wavelengths**(-1) + 
                                      c2*wavelengths**(-2) + 
                                      c3*wavelengths**(-3) + 
                                      c4*wavelengths**(-4))
                
                corrected_data[i] = (spectrum - scattering_component) / a
                
            except np.linalg.LinAlgError:
                # Fallback to basic EMSC if matrix is singular
                logger.warning(f"RMieS-EMSC failed for spectrum {i}, using basic EMSC")
                coeffs = np.polyfit(reference_spectrum, spectrum, 1)
                a, b = coeffs[0], coeffs[1]
                corrected_data[i] = (spectrum - b) / a
    
    logger.info(f"Applied RMieS-EMSC correction ({n_iterations} iterations)")
    return corrected_data

def me_emsc_correction(data, wavenumbers, reference=None, options=None,
                       return_residuals=False):
    """
    ME-EMSC (Mie Extinction EMSC) correction — Konevskikh et al. 2016.

    The proper Mie-scattering correction for IR spectra of cells and thin
    tissue. It builds an EMSC interferent basis from a database of Mie
    extinction curves Q_ext(nu; a, n) over a grid of sphere radii and
    refractive indices, then iteratively fits that basis out of each
    spectrum. This is what you want when you see scatter-induced baseline
    ripples or distorted amide I band shapes that vector normalisation
    cannot fix.

    Wraps the vendored port of Solheim/Konevskikh's BioSpecNorway ME-EMSC,
    based on Dougal Ferguson's `ME_EMSC_Python` (MIT). See
    `specml/processing/me_emsc/` for the algorithm itself.

    Args:
        data (np.ndarray): Spectral data, shape (n_samples, n_wavenumbers).
        wavenumbers (np.ndarray): Wavenumber axis (cm^-1).
        reference (np.ndarray, optional): Clean reference spectrum on the
            same wavenumber axis (e.g. Matrigel, casein, or a flat
            scatter-free tissue region). If None, the dataset mean is used
            — works, but a real reference is much better.
        options (dict, optional): ME-EMSC options forwarded to the underlying
            algorithm. Sensible keys (and defaults if omitted):
                'maxIterationNumber'  : 45
                'PCnumber'            : False  (auto from explained variance)
                'ExplainedVariance'   : 99.96
                'minRadius', 'maxRadius'             : 2, 7.1   (μm)
                'minRefractiveIndex', 'maxRefractiveIndex' : 1.1, 1.4
                'samplingSteps'       : 10
                'h'                   : 0.25
                'Weights'             : True
        return_residuals (bool): If True, also return RMSE residuals per
            spectrum (useful for quality-control filtering).

    Returns:
        np.ndarray: Corrected data, same shape as input. If
            `return_residuals=True`, returns `(corrected, rmse)`.
    """
    from specml.processing.me_emsc import ME_EMSC

    if reference is None:
        reference = np.mean(data, axis=0)
        logger.warning(
            "No reference provided for ME-EMSC; using dataset mean. "
            "Pass a clean reference (Matrigel, casein, etc.) for better results."
        )

    # ME-EMSC expects the reference as a (1, n_wn) row, spectra as (n, n_wn),
    # and wavenumbers as a 1D array.
    reference = np.asarray(reference).reshape(1, -1)
    data = np.asarray(data)
    wn = np.asarray(wavenumbers).flatten()

    # Min-to-zero on the reference so the iterative correction is well-posed.
    reference = reference - np.min(reference)

    opts = dict(options) if options else {}
    corrected, residuals, _params, _n_iters, _opts = ME_EMSC(
        reference, data, wn, opts
    )

    logger.info("Applied ME-EMSC correction")

    if return_residuals:
        rmse = np.sqrt(np.mean(residuals ** 2, axis=1))
        return corrected, rmse
    return corrected

def kmeans_tissue_wax_separation(data, wavenumbers, n_clusters=3, amide_range=(1600, 1700)):
    """
    K-means tissue/wax separation using integrated Amide I band.
    
    Following Ferguson's approach: use k-means on Amide I (1600-1700 cm⁻¹)
    to separate high-protein tissue pixels from wax/background.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        wavenumbers (np.ndarray): Wavenumber values
        n_clusters (int): Number of clusters for k-means (default: 3)
        amide_range (tuple): Wavenumber range for Amide I (default: 1600-1700)
    
    Returns:
        tuple: (tissue_mask, cluster_labels, amide_intensities)
    """
    from sklearn.cluster import KMeans
    from scipy.integrate import trapz
    
    # Find Amide I region indices
    amide_mask = (wavenumbers >= amide_range[0]) & (wavenumbers <= amide_range[1])
    amide_indices = np.where(amide_mask)[0]
    
    if len(amide_indices) == 0:
        logger.warning("No wavenumbers found in Amide I range")
        return np.ones(data.shape[0], dtype=bool), np.zeros(data.shape[0]), np.zeros(data.shape[0])
    
    # Calculate integrated Amide I intensity for each spectrum
    amide_data = data[:, amide_indices]
    amide_wavenumbers = wavenumbers[amide_indices]
    
    # Integrate using trapezoidal rule
    amide_intensities = np.array([trapz(spectrum, amide_wavenumbers) for spectrum in amide_data])
    
    # Reshape for k-means (needs 2D input)
    features = amide_intensities.reshape(-1, 1)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Identify tissue cluster (highest average Amide I intensity)
    cluster_means = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        if np.sum(cluster_mask) > 0:
            cluster_means.append(np.mean(amide_intensities[cluster_mask]))
        else:
            cluster_means.append(0)
    
    # Tissue cluster is the one with highest protein content
    tissue_cluster = np.argmax(cluster_means)
    tissue_mask = cluster_labels == tissue_cluster
    
    n_tissue = np.sum(tissue_mask)
    n_total = len(tissue_mask)
    logger.info(f"K-means tissue separation: {n_tissue}/{n_total} pixels classified as tissue ({n_tissue/n_total:.1%})")
    
    return tissue_mask, cluster_labels, amide_intensities

def debris_pruning(data, mask, retention_threshold=0.95):
    """
    Removes small tissue fragments and debris, keeping only the largest
    connected tissue regions until the specified retention threshold is met.
    
    Following Ferguson's approach: keep largest regions until ≥95% retention.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        mask (np.ndarray): Boolean mask indicating tissue pixels
        retention_threshold (float): Fraction of spectra to retain (default: 0.95)
    
    Returns:
        tuple: (pruned_data, pruned_mask)
    """
    from scipy import ndimage
    from sklearn.cluster import DBSCAN
    
    # Get spatial coordinates of tissue pixels
    tissue_indices = np.where(mask)[0]
    
    if len(tissue_indices) == 0:
        return data, mask
    
    # Simple spatial clustering approach
    # For proper implementation, would need spatial coordinates
    # This is a simplified version
    
    # Calculate connectivity based on spectral similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Use subset for clustering (computational efficiency)
    n_subset = min(10000, len(tissue_indices))
    subset_indices = np.random.choice(tissue_indices, n_subset, replace=False)
    subset_data = data[subset_indices]
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(subset_data)
    
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    
    # Cluster using DBSCAN
    clustering = DBSCAN(metric='precomputed', eps=0.3, min_samples=50)
    labels = clustering.fit_predict(distance_matrix)
    
    # Find largest clusters
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    
    if len(unique_labels) == 0:
        return data, mask
    
    # Sort clusters by size
    sorted_indices = np.argsort(counts)[::-1]
    sorted_labels = unique_labels[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Select clusters until retention threshold is met
    cumulative_count = 0
    selected_labels = []
    target_count = int(n_subset * retention_threshold)
    
    for label, count in zip(sorted_labels, sorted_counts):
        selected_labels.append(label)
        cumulative_count += count
        if cumulative_count >= target_count:
            break
    
    # Create new mask
    pruned_mask = mask.copy()
    selected_indices = []
    
    for label in selected_labels:
        cluster_indices = subset_indices[labels == label]
        selected_indices.extend(cluster_indices)
    
    # Update mask to keep only selected tissue regions
    keep_mask = np.zeros(len(tissue_indices), dtype=bool)
    for idx in selected_indices:
        pos = np.where(tissue_indices == idx)[0]
        if len(pos) > 0:
            keep_mask[pos[0]] = True
    
    final_tissue_indices = tissue_indices[keep_mask]
    pruned_mask = np.zeros(len(mask), dtype=bool)
    pruned_mask[final_tissue_indices] = True
    
    # Extract corresponding data
    pruned_data = data[final_tissue_indices]
    
    retention_rate = len(pruned_data) / len(data)
    logger.info(f"Debris pruning: retained {len(pruned_data)}/{len(data)} spectra ({retention_rate:.1%})")
    
    return pruned_data, pruned_mask

def robust_preprocessing_pipeline(data, wavenumbers, 
                                 remove_co2=True, 
                                 remove_water_vapor=True,
                                 baseline_method='als',
                                 normalization_method='snv',
                                 smoothing_window=7):
    """
    Comprehensive preprocessing pipeline for biological tissue FTIR data.
    
    Applies a series of preprocessing steps optimized for tissue analysis
    and lesion detection. Each step is scientifically motivated and
    preserves chemical information while removing artifacts.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        wavenumbers (np.ndarray): Wavenumber values
        remove_co2 (bool): Remove CO2 absorption regions
        remove_water_vapor (bool): Remove water vapor interference
        baseline_method (str): Baseline correction method
        normalization_method (str): Normalization method
        smoothing_window (int): Savitzky-Golay smoothing window
    
    Returns:
        tuple: (preprocessed_data, processing_log)
    """
    processing_log = []
    processed_data = data.copy()
    
    # 1. Remove atmospheric interferences
    if remove_co2:
        co2_regions = [(2300, 2400)]  # CO2 absorption region
        for start, end in co2_regions:
            region_mask = (wavenumbers >= start) & (wavenumbers <= end)
            region_indices = np.where(region_mask)[0]
            
            if len(region_indices) > 0:
                for i in range(processed_data.shape[0]):
                    left_idx = max(0, region_indices[0] - 5)
                    right_idx = min(len(wavenumbers) - 1, region_indices[-1] + 5)
                    
                    x_interp = [left_idx, right_idx]
                    y_interp = [processed_data[i, left_idx], processed_data[i, right_idx]]
                    
                    for j in region_indices:
                        processed_data[i, j] = np.interp(j, x_interp, y_interp)
        
        processing_log.append("Removed CO2 interference")
    
    if remove_water_vapor:
        processed_data = remove_water_vapor_interference(processed_data, wavenumbers)
        processing_log.append("Removed water vapor interference")
    
    # 2. Baseline correction
    if baseline_method == 'als':
        processed_data = als_baseline_correction(processed_data)
    elif baseline_method == 'rubberband':
        processed_data = rubberband_baseline_correction(processed_data)
    elif baseline_method == 'linear':
        processed_data = detrend_spectrum(processed_data, method='linear')
    
    processing_log.append(f"Applied {baseline_method} baseline correction")
    
    # 3. Normalization
    if normalization_method == 'snv':
        processed_data = snv_normalization(processed_data)
    elif normalization_method == 'msc':
        processed_data = msc_normalization(processed_data)
    elif normalization_method == 'emsc':
        processed_data = extended_multiplicative_scatter_correction(processed_data)
    
    processing_log.append(f"Applied {normalization_method} normalization")
    
    # 4. Smoothing
    if smoothing_window > 0:
        for i in range(processed_data.shape[0]):
            processed_data[i] = signal.savgol_filter(processed_data[i], 
                                                   smoothing_window, 2)
        processing_log.append(f"Applied Savitzky-Golay smoothing (window={smoothing_window})")
    
    return processed_data, processing_log