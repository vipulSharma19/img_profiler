import streamlit as st
import cv2


def detect_sift_features(img):
    """ Detect SIFT keypoints in an image using OpenCV.

    Args:
        img: Image data in numpy array format.

    Returns:
        keypoints: A list of keypoints detected in the image.
        descriptors: SIFT descriptors associated with the keypoints.
    """

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def visualize_keypoints(img):
    """ Visualize detected keypoints on the image.

    Args:
        img: Image data in numpy array format.
    """
    keypoints, descriptors = detect_sift_features(img)
    # Print the number of keypoints detected
    st.write("Number of keypoints detected:", len(keypoints))

    # Visualize the keypoints on the image

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(img_with_keypoints, channels="BGR", caption='Image with Keypoints')


