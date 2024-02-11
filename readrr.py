import numpy as np 
import matplotlib.pyplot as plt 

def read_img(filepath: str,gray:bool=False) -> np.ndarray:
    """
    Read an image file using matplotlib library and return it as a NumPy array.

    Parameters:
    - filepath (str): Path to the image file.

    Returns:
    - numpy.ndarray: Image represented as a 2D NumPy array.
    """
    img = plt.imread(filepath,)
    if gray:
        grayImage = np.zeros(img.shape)
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        R = (R *.299)
        G = (G *.587)
        B = (B *.114)

        Avg = (R+G+B)
        grayImage = img.copy()

        for i in range(3):
           grayImage[:,:,i] = Avg
           
        return grayImage      
    return img

def gray_img(img: np.ndarray) -> np.ndarray:
    grayImage = np.zeros((img.shape[0], img.shape[1]))  # Create a 2D array for the gray image
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * 0.299)  # Note the correction in coefficients
    G = (G * 0.587)
    B = (B * 0.114)

    Avg = R + G + B

    # Assign the calculated average to each channel separately
    grayImage[:, :] = Avg

    return grayImage