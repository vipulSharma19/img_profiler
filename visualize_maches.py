import cv2

def detect_sift_features(image_path):
    """ Detect SIFT keypoints in an image using OpenCV.

    Args:
        image_path (str): Path to the image file.

    Returns:
        keypoints: A list of keypoints detected in the image.
        descriptors: SIFT descriptors associated with the keypoints.
    """
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors

def visualize_keypoints(image_path, keypoints):
    """ Visualize detected keypoints on the image.

    Args:
        image_path (str): Path to the image file.
        keypoints: A list of keypoints detected in the image.
    """
    # Read the image
    img = cv2.imread(image_path)

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.namedWindow('Image with Keypoints', cv2.WINDOW_NORMAL)

    # Display the image with keypoints
    cv2.imshow('Image with Keypoints', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Path to the input image
    image_path = r'asd.jpg'

    # Detect SIFT features
    keypoints, descriptors = detect_sift_features(image_path)

    # Print the number of keypoints detected
    print("Number of keypoints detected:", len(keypoints))

    # Visualize the keypoints on the image
    visualize_keypoints(image_path, keypoints)

if __name__ == "__main__":
    main()
