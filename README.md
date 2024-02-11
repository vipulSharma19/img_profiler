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

