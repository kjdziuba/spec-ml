import numpy as np
import scipy.sparse
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.linalg import solveh_banded

def adjust_spectra_min_to_zero(data):
    """
    Adjusts each spectrum so that its minimum value is 0.
    """
    data_min = np.min(data, axis=-1, keepdims=True)
    return data - data_min


# VECTOR NORM BORROWED AND ADJUSTED FROM DOUGAL FERGUSON'S SPECTRALCOLLECTION
def vector_norm(data=None):
        """Vector normalises the object's spectral dataset along each 
        wavenumber measurement point.
        
        :param data: 2D dataset to vector normalise. Default = 0
        :type data: array of float64.

        :returns: numpy.array
        
        """
        
        is_sparseinput = scipy.sparse.issparse(data)
        
        squares = np.square(data)   # square of each variable ([n,m])
        sum_of_squares = np.sum(squares, axis = 1)   #sum of the squares along the rows ([1,n]) 
        divisor = np.sqrt(sum_of_squares)    # ([n,1]) == L2-Norm
        divisor[divisor == 0] = 1 #avoid div/0 error
        #Make data sparse matrix for memory 
        
        # Generate a sparse matrix where the diagonal is of the inverse of the divisor
        multiplier = 1./ divisor
        
        #delete items for memory 
        del squares
        del sum_of_squares
        del divisor
        
        multiplierdiag = scipy.sparse.csr_matrix(scipy.sparse.diags(multiplier))
        
        # multiply the two matrices and then convert data back to a dense matrix
        # which is dividing the data by the vector length 
        data = multiplierdiag * data
        #self.data = self.data.todense()
        
        del multiplierdiag
        #self.data = np.matmul(self.data,multiplierdiag) # divide the data by the vector length ([n,m])
        
        if is_sparseinput == True:
            data = scipy.sparse.csr_matrix(data)

        return data

def apply_savitzky_golay(data, wavenumbers, window_length=11, polyorder=2, deriv=0, interp=False):
    """
    Applies Savitzky-Golay smoothing and/or derivative to the spectral data.

    :param data: The 2D spectral dataset.
    :param window_length: Length of the filter window (must be odd). Default is 11.
    :param polyorder: The order of the polynomial used to fit the samples. Default is 2.
    :param deriv: Derivative order to compute. Default is 1 (first derivative).
    :returns: Smoothed and/or differentiated data.
    """
     # Ensure window_length is less than or equal to the number of wavenumbers
    if window_length > data.shape[-1]:
        raise ValueError("window_length must be less than or equal to the number of data points (wavenumbers).")
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1  # Make it odd by adding 1

    if interp:
        # Use interpolation to handle edges
        processed_data = savgol_filter(data, window_length, polyorder, deriv=deriv, axis=-1, mode='interp')
        new_wavenumbers = wavenumbers  # Wavenumbers remain the same
    else:
        # Apply the filter without interpolation
        processed_data = savgol_filter(data, window_length, polyorder, deriv=deriv, axis=-1)

        # Calculate the number of points to trim from each end
        deletion_point = int((window_length - 1) / 2)

        # Trim the data and wavenumbers to remove edge effects
        processed_data = processed_data[:, deletion_point:-deletion_point]
        new_wavenumbers = wavenumbers[deletion_point:-deletion_point]
    return processed_data, new_wavenumbers

def apply_gaussian_smoothing(data, sigma=1):
    """
    Applies Gaussian smoothing to the spectral data.

    :param data: The spectral dataset.
    :param sigma: Standard deviation for Gaussian kernel. Default is 1.
    :returns: Smoothed data.
    """
    smoothed_data = gaussian_filter(data, sigma=sigma)
    return smoothed_data


def mean_center(data):
    """
    Mean-centers the spectral data by subtracting the mean spectrum.

    :param data: The spectral dataset.
    :returns: Mean-centered data.
    """

    mean_spectrum = np.mean(data, axis=0)
    centered_data = data - mean_spectrum
    return centered_data


def baseline_correction(data, lam=10000, p=0.01, niter=10):
    """
    Performs baseline correction using an asymmetric least squares method.

    :param data: The spectral dataset.
    :param lam: Smoothing parameter.
    :param p: Asymmetry parameter.
    :param niter: Number of iterations.
    :returns: Baseline-co   rrected data.
    """
    def baseline(y, lam, p, niter):
        L = len(y)
        D = np.diff(np.eye(L), 2)
        w = np.ones(L)
        for _ in range(niter):
            W = np.diag(w)
            Z = W + lam * D.T @ D
            z = np.linalg.solve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    corrected_data = np.array([y - baseline(y, lam, p, niter) for y in data])
    return corrected_data

def z_score_normalize(data):
    """
    Normalizes the data using Z-score (Standard Score) normalization.

    :param data: The spectral dataset.
    :returns: Z-score normalized data.
    """
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return data


def split_into_squares(data, square_size=(128, 128)):
    """
    Splits the 3D data into smaller squares of the given size (e.g., 128x128).
    
    :param data: The spectral dataset.
    :param square_size: Tuple specifying the size of each square.
    :returns: List of square segments.
    """
    ypixels, xpixels, wavenumbers = data.shape
    squares = []
    for y in range(0, ypixels, square_size[0]):
        for x in range(0, xpixels, square_size[1]):
            square = data[y:y+square_size[0], x:x+square_size[1], :]
            squares.append(square)
    return squares

def stitch_squares(squares, original_shape):
    """
    Stitches square segments back into the original 3D shape.
    
    :param squares: List of square segments.
    :param original_shape: Tuple representing the original shape (ypixels, xpixels, wavenumbers).
    :returns: Reconstructed 3D data.
    """
    ypixels, xpixels, wavenumbers = original_shape
    data = np.zeros(original_shape)
    square_size = squares[0].shape[:2]
    index = 0
    for y in range(0, ypixels, square_size[0]):
        for x in range(0, xpixels, square_size[1]):
            data[y:y+square_size[0], x:x+square_size[1], :] = squares[index]
            index += 1
    return data

def pca_denoising(data, n_components=10):
    """
    Performs PCA-based denoising on spectral data.

    :param data: The spectral dataset (NumPy array of shape [n_samples, n_wavenumbers]).
    :param n_components: Number of principal components to retain. Default is 10.
    :returns: Denoised data reconstructed from the top principal components.
    """
    from sklearn.decomposition import PCA

    # Ensure n_components is less than the number of wavenumbers
    if n_components >= data.shape[1]:
        raise ValueError("n_components must be less than the number of wavenumbers.")

    # Center the data by subtracting the mean spectrum
    mean_spectrum = np.mean(data, axis=0)
    data_centered = data - mean_spectrum

    # Fit PCA on the centered data
    pca = PCA(n_components=n_components)
    pca.fit(data_centered)

    # Transform data into principal component space
    transformed_data = pca.transform(data_centered)

    # Reconstruct the data from the selected principal components
    reconstructed_data = pca.inverse_transform(transformed_data)

    # Add the mean back to the reconstructed data
    denoised_data = reconstructed_data + mean_spectrum

    return denoised_data

def mnf_denoising(data, n_components=10):
    """
    Performs MNF-based denoising on spectral data.

    :param data: The spectral dataset (NumPy array of shape [n_samples, n_wavenumbers]).
    :param n_components: Number of MNF components to retain. Default is 10.
    :returns: Denoised data reconstructed from the top MNF components.
    """
    from scipy.linalg import fractional_matrix_power
    from scipy.ndimage import gaussian_filter1d

    # Estimate the noise covariance matrix
    # Assume noise is the difference between the data and its smoothed version
    smoothed_data = gaussian_filter1d(data, sigma=1, axis=1)
    noise = data - smoothed_data
    noise_cov = np.cov(noise, rowvar=False)

    # Estimate the signal covariance matrix
    signal_cov = np.cov(data, rowvar=False)

    # Check for positive-definite noise covariance matrix
    if not np.all(np.linalg.eigvals(noise_cov) > 0):
        raise ValueError("Noise covariance matrix is not positive-definite.")

    # Perform the noise whitening transform
    # Compute inverse square root of noise covariance matrix
    noise_cov_inv_sqrt = fractional_matrix_power(noise_cov, -0.5)

    # Whiten the signal covariance matrix
    whitened_signal_cov = noise_cov_inv_sqrt @ signal_cov @ noise_cov_inv_sqrt

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(whitened_signal_cov)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    eigenvectors = eigenvectors[:, :n_components]

    # Compute the transformation matrix
    transformation_matrix = noise_cov_inv_sqrt @ eigenvectors

    # Transform data into MNF space
    mnf_scores = data @ transformation_matrix

    # Reconstruct data from MNF components
    reconstructed_data = mnf_scores @ transformation_matrix.T

    return reconstructed_data