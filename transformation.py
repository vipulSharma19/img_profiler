import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsv(r: float, g: float, b: float) -> tuple:
    """
    Convert RGB values to HSV.

    Parameters:
    - r (float): Red component (0 to 255).
    - g (float): Green component (0 to 255).
    - b (float): Blue component (0 to 255).

    Returns:
    - tuple: HSV values (hue, saturation, value).
    """
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin

    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100

    v = cmax * 100
    return h, s, v

def to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to the HSV color space.

    Parameters:
    - image (np.ndarray): Input RGB image.

    Returns:
    - np.ndarray: HSV image.

    """
    image = (image * 255).astype(np.uint8)
    height, width, channels = image.shape
    hsv_image = np.zeros_like(image, dtype=np.float64)

    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]  # Get RGB values for each pixel
            hsv_image[i, j] = rgb_to_hsv(r, g, b)  # Convert RGB to HSV
    # Display the original and HSV images
    plt.subplot(1, 2, 1), plt.imshow(image), plt.title('Original Image')
    plt.subplot(1, 2, 2), plt.imshow(hsv_image / 255.0), plt.title('HSV Image')
    plt.show()
    return hsv_image

