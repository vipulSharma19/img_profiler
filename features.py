import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, filters, io
import numpy as np
from scipy.stats import kurtosis, skew
def sharpness(array: np.ndarray) -> float:
    """
    Compute the sharpness of an image represented by a 2D array.

    Parameters:
    - array (numpy.ndarray): Input image as a 2D numpy array.

    Returns:
    - float: Sharpness value computed as the average gradient magnitude of the image.
    """
    # Compute the gradient using numpy.gradient
    gradients = np.gradient(array)

    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(np.sum(np.square(gradients), axis=0))

    # Compute sharpness as the average gradient magnitude
    sharpness = np.average(gradient_magnitude)

    return sharpness



def lbpv(image: np.ndarray, radius: int = 1, neighbors: int = 8) -> np.ndarray:
    """
    Compute Local Binary Pattern Variance (LBPV) for a given image.

    Parameters:
    - image (numpy.ndarray): Input image as a 2D NumPy array.
    - radius (int): Radius of the circular neighborhood around each pixel.
    - neighbors (int): Number of neighbors considered for LBP computation.

    Returns:
    - float: Variance of the LBP image computed from the input image.
    """
    # Compute LBP
    lbp = np.uint8(np.zeros_like(image))
    for i in range(len(image)):
        for j in range(len(image[0])):
            center = image[i, j]
            pattern = 0
            for k in range(neighbors):
                x = i + int(radius * np.cos(2 * np.pi * k / neighbors))
                y = j - int(radius * np.sin(2 * np.pi * k / neighbors))
                if x >= 0 and x < len(image) and y >= 0 and y < len(image[0]):
                    pattern |= (image[x, y] >= center) << (neighbors - k - 1)
            lbp[i, j] = pattern
        # Compute LBP variance
    lbp_variance = np.var(lbp)
    
    return lbp_variance

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

#
# def lpq(img: np.ndarray, winSize: int = 3, freqestim: int = 1, mode: str = 'nh') -> np.ndarray:
#     """
#     Compute LPQ (Local Phase Quantization) descriptor for the given image.
#
#     Parameters:
#     - img (np.ndarray): Input image (2D numpy array).
#     - winSize (int): Size of the window for computing LPQ. Default is 3.
#     - freqestim (int): Frequency estimation method:
#         - 1: STFT uniform window (default).
#     - mode (str): Output mode:
#         - 'nh': Normalized histogram (default).
#         - 'h': Histogram.
#         - 'im': LPQ code image.
#
#     Returns:
#     - LPQdesc (np.ndarray): LPQ descriptor.
#
#     Raises:
#     - ValueError: If an unsupported frequency estimation method or mode is provided.
#     """
#     rho = 0.90
#     STFTalpha = 1 / winSize
#     sigmaS = (winSize - 1) / 4
#     sigmaA = 8 / (winSize - 1)
#     convmode = 'valid'
#
#     img = np.float64(img)
#     r = (winSize - 1) / 2
#     x = np.arange(-r, r + 1)[np.newaxis]
#
#     if freqestim == 1:
#         w0 = np.ones_like(x)
#         w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
#         w2 = np.conj(w1)
#
#     filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
#     filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
#     filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
#     filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)
#
#     freqResp = np.dstack([filterResp1.real, filterResp1.imag,
#                          filterResp2.real, filterResp2.imag,
#                          filterResp3.real, filterResp3.imag,
#                          filterResp4.real, filterResp4.imag])
#
#     inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
#     LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)
#
#     if mode == 'im':
#         LPQdesc = np.uint8(LPQdesc)
#         plt.imshow(LPQdesc, cmap='gray')
#         plt.title("LPQ Code Image")
#         plt.axis('off')
#         plt.show()
#
#     if mode == 'nh' or mode == 'h':
#         LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]
#
#         if mode == 'nh':
#             LPQdesc = LPQdesc / LPQdesc.sum()
#
#         plt.bar(np.arange(len(LPQdesc)), LPQdesc)
#         plt.title("LPQ Histogram")
#         plt.xlabel("Bin")
#         plt.ylabel("Frequency")
#         plt.show()
#
#     return LPQdesc
import streamlit as st
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image


def lpq(img: np.ndarray, winSize: int = 3, freqestim: int = 1, mode: str = 'nh') -> np.ndarray:
    """
    Compute LPQ (Local Phase Quantization) descriptor for the given image.

    Parameters:
    - img (np.ndarray): Input image (2D numpy array).
    - winSize (int): Size of the window for computing LPQ. Default is 3.
    - freqestim (int): Frequency estimation method:
        - 1: STFT uniform window (default).
    - mode (str): Output mode:
        - 'nh': Normalized histogram (default).
        - 'h': Histogram.
        - 'im': LPQ code image.

    Returns:
    - LPQdesc (np.ndarray): LPQ descriptor.

    Raises:
    - ValueError: If an unsupported frequency estimation method or mode is provided.
    """
    rho = 0.90
    STFTalpha = 1 / winSize
    sigmaS = (winSize - 1) / 4
    sigmaA = 8 / (winSize - 1)
    convmode = 'valid'

    img = np.float64(img)
    r = (winSize - 1) / 2
    x = np.arange(-r, r + 1)[np.newaxis]

    if freqestim == 1:
        w0 = np.ones_like(x)
        w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
        w2 = np.conj(w1)

    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)

    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                          filterResp2.real, filterResp2.imag,
                          filterResp3.real, filterResp3.imag,
                          filterResp4.real, filterResp4.imag])

    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)

    if mode == 'im':
        LPQdesc = np.uint8(LPQdesc)
        plt.imshow(LPQdesc, cmap='gray')
        plt.title("LPQ Code Image")
        plt.axis('off')
        st.pyplot()

    if mode == 'nh' or mode == 'h':
        LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]

        if mode == 'nh':
            LPQdesc = LPQdesc / LPQdesc.sum()

        plt.bar(np.arange(len(LPQdesc)), LPQdesc)
        plt.title("LPQ Histogram")
        plt.xlabel("Pixels")
        plt.ylabel("Frequency")
        st.pyplot()

    return LPQdesc





def fractal_dimension(image_data, max_box_size=None, min_box_size=1, n_samples=20, n_offsets=0, plot=False):
    """
    Calculates the fractal dimension of a 2D or 3D image.

    Args:
        image (np.ndarray): The image to calculate the fractal dimension of.
        max_box_size (int): The largest box size, given as the power of 2 so that
                            2**max_box_size gives the sidelength of the largest box.
        min_box_size (int): The smallest box size, given as the power of 2 so that
                            2**min_box_size gives the sidelength of the smallest box.
                            Default value 1.
        n_samples (int): Number of scales to measure over.
        n_offsets (int): Number of offsets to search over to find the smallest set N(s) to
                         cover all pixels>0.
        plot (bool): Set to True to see the analytical plot of a calculation.
    """
    image_channel = image_data[:, :, 0]  # Assuming the image is RGB and you want to work with only one channel
    gamma_adjusted_image = exposure.adjust_gamma(image_channel, gamma=1.2)
    smoothed_image = filters.gaussian(gamma_adjusted_image, sigma=0.8)
    binary_image = smoothed_image > filters.threshold_otsu(smoothed_image)
    image=binary_image
    if max_box_size is None:
        max_box_size = int(np.floor(np.log2(np.min(image.shape))))

    scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_samples, base=2))
    scales = np.unique(scales)

    # Get the locations of all non-zero pixels
    nonzero_pixel_coords = np.where(image > 0)

    if image.ndim == 2:
        # 2D case
        voxel_coords = np.array([(x, y) for x, y in zip(*nonzero_pixel_coords)])
    elif image.ndim == 3:
        # 3D case
        voxel_coords = np.array([(x, y, z) for x, y, z in zip(*nonzero_pixel_coords)])
    else:
        raise ValueError("Unsupported image dimension. Must be 2D or 3D.")

    # Count the minimum amount of boxes touched
    touched_boxes = []
    # Loop over all scales
    for scale in scales:
        touched_offsets = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)
        # Search over all offsets
        for offset in offsets:
            bin_edges = [np.arange(0, i, scale) for i in image.shape]
            bin_edges = [np.hstack([0 - offset, x + offset]) for x in bin_edges]
            hist, edges = np.histogramdd(voxel_coords, bins=bin_edges)
            touched_offsets.append(np.sum(hist > 0))
        touched_boxes.append(touched_offsets)

    touched_boxes = np.array(touched_boxes)

    # From all sets N found, keep the smallest one at each scale
    touched_boxes_min = touched_boxes.min(axis=1)

    # Only keep scales at which Ns changed
    scales_changed = np.array([np.min(scales[touched_boxes_min == x]) for x in np.unique(touched_boxes_min)])
    touched_boxes_min = np.unique(touched_boxes_min)
    touched_boxes_min = touched_boxes_min[touched_boxes_min > 0]
    scales_changed = scales_changed[:len(touched_boxes_min)]

    # Perform fit
    fit_coeffs = np.polyfit(np.log(1/scales_changed), np.log(touched_boxes_min), 1)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.log(1 / scales_changed), np.log(np.unique(touched_boxes_min)), c="teal", label="Measured ratios")
        ax.set_ylabel("$\log N(\epsilon)$")
        ax.set_xlabel("$\log 1/ \epsilon$")
        fitted_y_vals = np.polyval(fit_coeffs, np.log(1 / scales_changed))
        ax.plot(np.log(1 / scales_changed), fitted_y_vals, "k--",
                label=f"Fit: {np.round(fit_coeffs[0], 3)}X+{fit_coeffs[1]}")
        ax.legend()
        st.pyplot()

    return fit_coeffs[0]



def texture_analysis(gray):
    # Convert the image to grayscale

    # Compute gradients using Sobel operators
    dx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    dy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    grad_x = convolve2d(gray, dx, mode='same', boundary='symm')
    grad_y = convolve2d(gray, dy, mode='same', boundary='symm')

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Compute statistics
    mean_gradient = np.mean(gradient_magnitude)
    std_dev_gradient = np.std(gradient_magnitude)
    kurt_gradient = kurtosis(gradient_magnitude.flatten())
    skew_gradient = skew(gradient_magnitude.flatten())

    # Return texture analysis dictionary
    texture = {
        "mean": mean_gradient,
        "std_dev": std_dev_gradient,
        "kurtosis": kurt_gradient,
        "skewness": skew_gradient
    }
    return texture
