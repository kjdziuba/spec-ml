"""
Quality control functions for FTIR spectroscopic data preprocessing.
Essential for biological tissue analysis and lesion detection.
"""

import numpy as np
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance

logger = logging.getLogger(__name__)

def remove_saturated_spectra(data, wavenumbers, threshold=0.95, max_intensity=4.0):
    """
    Removes spectra that show signs of detector saturation.
    
    Critical for FTIR data quality in biological samples where water absorption
    can cause saturation around 1640 cm⁻¹ and 3200-3600 cm⁻¹.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers) or (batch, n_samples, n_wavenumbers)
        wavenumbers (np.ndarray): Wavenumber values
        threshold (float): Fraction of max intensity to consider saturated
        max_intensity (float): Maximum expected absorbance value
    
    Returns:
        tuple: (clean_data, valid_indices)
    """
    # Handle different data shapes from SpectroscopicData
    if data.ndim == 3:
        # Shape is (batch, n_samples, n_wavenumbers) - reshape to (n_samples, n_wavenumbers)
        original_shape = data.shape
        data = data.reshape(-1, data.shape[-1])
    
    max_values = np.max(data, axis=1)
    valid_mask = max_values < (threshold * max_intensity)
    
    n_removed = np.sum(~valid_mask)
    if n_removed > 0:
        logger.warning(f"Removed {n_removed} saturated spectra ({n_removed/len(data)*100:.1f}%)")
    
    return data[valid_mask], valid_mask

# Cosmic ray removal removed - not relevant for biological FTIR analysis

def detect_outlier_spectra(data, method='mahalanobis', threshold=3.0):
    """
    Detects outlier spectra using statistical methods.
    
    Essential for biological samples where contamination, air bubbles,
    or sample preparation artifacts can create outlier spectra.
    
    Args:
        data (np.ndarray): Spectral data (n_samples, n_wavenumbers)
        method (str): 'mahalanobis', 'zscore', or 'isolation_forest'
        threshold (float): Threshold for outlier detection
    
    Returns:
        np.ndarray: Boolean mask of outlier spectra
    """
    if method == 'mahalanobis':
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Calculate Mahalanobis distance
        cov = EmpiricalCovariance().fit(scaled_data)
        mahal_distances = cov.mahalanobis(scaled_data)
        
        # Use chi-squared distribution for threshold
        threshold_value = stats.chi2.ppf(1 - 0.01, df=data.shape[1])  # 99% confidence
        outlier_mask = mahal_distances > threshold_value
        
    elif method == 'zscore':
        # Calculate Z-score for each spectrum
        mean_spectrum = np.mean(data, axis=0)
        std_spectrum = np.std(data, axis=0)
        z_scores = np.abs((data - mean_spectrum) / std_spectrum)
        
        # Flag spectra with high average Z-score
        avg_z_scores = np.mean(z_scores, axis=1)
        outlier_mask = avg_z_scores > threshold
        
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        
        # Use PCA to reduce dimensionality for outlier detection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, data.shape[1]))
        pca_data = pca.fit_transform(data)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(pca_data)
        outlier_mask = outlier_labels == -1
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    n_outliers = np.sum(outlier_mask)
    logger.info(f"Detected {n_outliers} outlier spectra ({n_outliers/len(data)*100:.1f}%)")
    
    return outlier_mask

def validate_preprocessing_quality(data, wavenumbers, checks=None):
    """
    Validates the quality of preprocessed FTIR data.
    
    Critical for ensuring preprocessing hasn't introduced artifacts
    that could affect downstream analysis.
    
    Args:
        data (np.ndarray): Preprocessed spectral data
        wavenumbers (np.ndarray): Wavenumber values
        checks (list): List of validation checks to perform
    
    Returns:
        dict: Dictionary with validation results
    """
    if checks is None:
        checks = ['nan_inf', 'intensity_range', 'spectral_quality', 'baseline_drift']
    
    results = {}
    
    # Check for NaN or infinite values
    if 'nan_inf' in checks:
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))
        results['nan_inf'] = {
            'nan_count': nan_count,
            'inf_count': inf_count,
            'passed': nan_count == 0 and inf_count == 0
        }
    
    # Check intensity range
    if 'intensity_range' in checks:
        min_val = np.min(data)
        max_val = np.max(data)
        # After normalization, values should be reasonable
        reasonable_range = -5 < min_val < 5 and -5 < max_val < 5
        results['intensity_range'] = {
            'min_value': min_val,
            'max_value': max_val,
            'passed': reasonable_range
        }
    
    # Check spectral quality (signal-to-noise ratio)
    if 'spectral_quality' in checks:
        # Calculate SNR in fingerprint region (1000-1800 cm⁻¹)
        fingerprint_indices = np.where((wavenumbers >= 1000) & (wavenumbers <= 1800))[0]
        if len(fingerprint_indices) > 0:
            fingerprint_data = data[:, fingerprint_indices]
            signal = np.mean(fingerprint_data, axis=1)
            noise = np.std(fingerprint_data, axis=1)
            snr = signal / (noise + 1e-10)  # Avoid division by zero
            avg_snr = np.mean(snr)
            results['spectral_quality'] = {
                'avg_snr': avg_snr,
                'passed': avg_snr > 5.0  # Minimum acceptable SNR
            }
    
    # Check for baseline drift
    if 'baseline_drift' in checks:
        # Check if baseline is approximately flat
        baseline_regions = [
            (wavenumbers >= 1900) & (wavenumbers <= 2000),  # Quiet region
            (wavenumbers >= 2100) & (wavenumbers <= 2200)   # Another quiet region
        ]
        
        drift_scores = []
        for region_mask in baseline_regions:
            region_indices = np.where(region_mask)[0]
            if len(region_indices) > 10:
                region_data = data[:, region_indices]
                # Calculate standard deviation across wavenumbers (should be low)
                drift = np.mean(np.std(region_data, axis=1))
                drift_scores.append(drift)
        
        if drift_scores:
            avg_drift = np.mean(drift_scores)
            results['baseline_drift'] = {
                'avg_drift': avg_drift,
                'passed': avg_drift < 0.1  # Acceptable drift level
            }
    
    # Overall validation
    all_passed = all(result.get('passed', True) for result in results.values())
    results['overall_passed'] = all_passed
    
    return results

def calculate_data_statistics(data, wavenumbers):
    """
    Calculates comprehensive statistics for FTIR data quality assessment.
    
    Args:
        data (np.ndarray): Spectral data
        wavenumbers (np.ndarray): Wavenumber values
    
    Returns:
        dict: Statistics dictionary
    """
    stats_dict = {
        'n_spectra': data.shape[0],
        'n_wavenumbers': data.shape[1],
        'wavenumber_range': (np.min(wavenumbers), np.max(wavenumbers)),
        'wavenumber_resolution': np.mean(np.diff(wavenumbers)),
        'intensity_stats': {
            'min': np.min(data),
            'max': np.max(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data)
        },
        'spectral_variance': np.var(data, axis=0),
        'mean_spectrum': np.mean(data, axis=0),
        'std_spectrum': np.std(data, axis=0)
    }
    
    return stats_dict


def calculate_snr(spectrum, wavenumbers, signal_region=(1600, 1700), noise_region=(1900, 2000)):
    """
    Calculate signal-to-noise ratio for a single spectrum.
    
    Args:
        spectrum (np.ndarray): Single spectrum (1D array)
        wavenumbers (np.ndarray): Wavenumber values
        signal_region (tuple): Wavenumber range for signal calculation (default: Amide I region)
        noise_region (tuple): Wavenumber range for noise calculation (default: quiet region)
    
    Returns:
        float: Signal-to-noise ratio
    """
    # Find indices for signal region
    signal_mask = (wavenumbers >= signal_region[0]) & (wavenumbers <= signal_region[1])
    signal_data = spectrum[signal_mask]
    
    # Find indices for noise region
    noise_mask = (wavenumbers >= noise_region[0]) & (wavenumbers <= noise_region[1])
    noise_data = spectrum[noise_mask]
    
    # Calculate SNR
    if len(signal_data) > 0 and len(noise_data) > 0:
        signal = np.mean(np.abs(signal_data))
        noise = np.std(noise_data)
        snr = signal / (noise + 1e-10)  # Avoid division by zero
    else:
        snr = 0.0
        logger.warning("Signal or noise region not found in wavenumber range")
    
    return snr