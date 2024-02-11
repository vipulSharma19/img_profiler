# ImageProfiler - A Python Library for Image Analysis and Report Generation


ImageProfiler is a Python library designed to offer in-depth analysis and reporting for image datasets. Inspired by the functionality of libraries such as pandas-profiling but focused on image data, ImageProfiler provides intuitive tools to extract meaningful insights from images, making it an essential tool for data scientists, researchers, and developers working in image processing, computer vision, and related fields.


## Installation

You can install ImageProfiler using pip:

```bash
pip install imgProfiler
```

## Contributing
We welcome contributions to ImageProfiler! Please read our contributing guidelines for more information.

## License
ImageProfiler is licensed under the MIT License.

## Acknowledgments

ImageProfiler was created by the team itertools at HackthisFall 2024. We're passionate about open source and are excited to contribute to the community with tools that help make data analysis more accessible and insightful.

For any questions or feedback, please open an issue on our GitHub repository.



# Functions provided

**read_img()**

    Reads an image from the specified file path and optionally converts it to grayscale.
    
    Parameters:
        filepath (str): The path to the image file.
        gray (bool, optional): If set to True, converts the image to grayscale. Default is False.
    
    Returns:
        np.ndarray: The image as a NumPy array. If gray is True, returns a grayscale image, otherwise returns the image in its original color.


**gray_img()**

    Converts an RGB image to a grayscale image.
    
    Parameters:
        img (np.ndarray): An RGB image as a NumPy array.
    
    Returns:

    np.ndarray: The grayscale version of the input image.

**lbpv()**

    Computes the Local Binary Pattern Variance (LBPV) for an input image, offering a measure of texture variance.
    
    Parameters:
    
        image (numpy.ndarray): The input image as a 2D NumPy array. This should be a grayscale image.
        radius (int, optional): The radius defining the circular neighborhood around each pixel for LBP computation. Defaults to 1.
        neighbors (int, optional): The number of equidistant neighbors considered around each pixel in the defined radius. Defaults to 8.
    
    Returns:
    
        float: The variance of the LBP image computed from the input image, representing the texture variance.

**lpq()**
    
    Calculates the LPQ descriptor for the given image, offering different output modes including the normalized histogram, histogram, and LPQ code image.
    Parameters
    
        img (np.ndarray): The input image as a 2D numpy array.
        winSize (int): The size of the window for computing LPQ. Defaults to 3.
        freqestim (int): The frequency estimation method. Currently, only 1 (STFT uniform window) is supported. Defaults to 1.
        mode (str): The output mode, which can be:
            'nh': Normalized histogram (default).
            'h': Histogram.
            'im': LPQ code image.
    
    Returns
    
        LPQdesc (np.ndarray): The LPQ descriptor, format depending on the selected mode.

**texture_analysis()**

    Analyzes the texture of a given grayscale image by calculating its gradient magnitude and deriving statistical measures from it.
    Parameters
    
        gray (np.ndarray): Input image in grayscale format. It must be a 2D numpy array.
    
    Returns
    
        dict: A dictionary containing statistical measures of the texture, including the mean, standard deviation, kurtosis, and skewness of the gradient magnitude.
        

**fractal_dimension()**

    Estimates the fractal dimension of an image using the box-counting method, which involves covering the image with a grid of boxes and counting how many boxes contain a part of the image's structure.
    Parameters
    
        image_data (np.ndarray): The image (2D or 3D) to calculate the fractal dimension of. It should be a numpy array.
        max_box_size (int, optional): The largest box size as the power of 2. Defaults to the power of 2 that is just smaller than the smallest image dimension.
        min_box_size (int, optional): The smallest box size as the power of 2. Defaults to 1.
        n_samples (int, optional): The number of scales (box sizes) to measure over. Defaults to 20.
        n_offsets (int, optional): The number of offsets to try for each box size, to minimize the number of boxes needed. Defaults to 0 (no offset).
        plot (bool, optional): If True, plots the relationship between the scale (box size) and the number of boxes needed. Useful for visualizing the fractal nature of the image. Defaults to False.
    
    Returns
    
        float: The estimated fractal dimension of the image.

**sharpness()**
    

    array (numpy.ndarray): The input image represented as a 2D NumPy array. The input image should be in grayscale format.

    Returns:

    float: The sharpness value of the input image. A higher value indicates a sharper image.
