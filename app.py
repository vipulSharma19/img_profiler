import streamlit as st
import symmetry
from symmetry import bilateral_syemmetry, test_case
import features
import readrr
from readrr import gray_img
import logging
from sift_keypoints import visualize_keypoints
import numpy as np

# logging.basicConfig(level=logging.ERROR)
# st.set_option('deprecation.showPyplotGlobalUse', False)
# def main():
#     st.title("Symmetry Detection App")
#     image_path = r"C:\Users\sharm\Downloads\10 (8).png"
#
#     st.subheader(f"Mirror Line Graph for image")
#     fig = bilateral_syemmetry(image_path, "With Mirror Line")
#     st.pyplot(fig)
#
#     image = readrr.read_img(image_path, gray=False)  # Use read_img or gray_img as needed
#     image2=readrr.gray_img(image)
#
#     #Compute and display additional features
#     st.subheader('Image Sharpness')
#     st.subheader(features.sharpness(image))
#     st.write("LBPV:", features.lbpv(image))
#     mylpq = features.lpq(image2, mode='h')
#         # st.write("LPQ:")  # Display LPQ code image   # Display LPQ histogram
#     st.markdown('*Fractal Dimension*')
#     st.write("Fractal Dimension:", features.fractal_dimension(image, plot=True))
#     st.write("Texture Analysis:", features.texture_analysis(image2))
#     visualize_keypoints(image)
logging.basicConfig(level=logging.ERROR)
st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
    st.title("imgProfiler")
    st.subheader("by Itertools")
    image_path = r"C:\Users\sharm\Downloads\10 (8).png"

    st.subheader(f"Mirror Line Graph for image")
    fig = bilateral_syemmetry(image_path, "With Mirror Line")
    st.pyplot(fig)

    image = readrr.read_img(image_path, gray=False)
    if image is None or len(image) == 0:
        st.error("Error loading the image.")
        return
    # Use read_img or gray_img as needed
    image3 = (image * 255).astype(np.uint8)

    image2 = readrr.gray_img(image)

    # Compute and display additional features
    st.subheader('Image Sharpness')
    st.subheader(features.sharpness(image))
    #st.write("LBPV:", features.lbpv(image))
    mylpq = features.lpq(image2, mode='h')
    # st.write("LPQ:")  # Display LPQ code image   # Display LPQ histogram
    st.markdown('*Fractal Dimension*')
    st.write("Fractal Dimension:", features.fractal_dimension(image, plot=True))
    st.write("Texture Analysis:", features.texture_analysis(image2))

    # Call visualize_keypoints() and show SIFT keypoints
    visualize_keypoints(image3)


if __name__ == "__main__":
    main()