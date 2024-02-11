
from streamlit_option_menu import option_menu
import streamlit as st
import symmetry
from symmetry import bilateral_syemmetry, test_case
import features
import readrr
from readrr import gray_img
import logging
from skimage import color, filters`1`   `
from sift_keypoints import visualize_keypoints
import numpy as np
import logging
import streamlit as st
import cv2
import numpy as np
from skimage import color, filters
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Json"],
        icons=["house", "gear", ],
        menu_icon="cast",
        default_index=0,

        # orientation = "horizontal",
    )
image_path = r"C:\Users\sharm\Downloads\10 (8).png"
logging.basicConfig(level=logging.ERROR)
st.set_option('deprecation.showPyplotGlobalUse', False)


def calculate_image_noise(img):
    noise = np.std(img)
    return noise

def calculate_exposure_color(img):
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y_channel = yuv_img[:,:,0]
    exposure = y_channel.mean()
    return exposure

def calculate_fluency_feature(img, weight_intensity=0.5, weight_gradient=0.5):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance_intensity = grayscale_img.var()
    gradient_magnitude = filters.sobel(grayscale_img).var()
    fluency_feature = weight_intensity * variance_intensity + weight_gradient * gradient_magnitude
    return fluency_feature
if selected =='Home':
    st.markdown("<h1 style='text-align: center; color: white;'>Image Profiling Tool</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>By - Itertools </h2>", unsafe_allow_html=True)




    image = readrr.read_img(image_path, gray=False)
    if image is None or len(image) == 0:
        st.error("Error loading the image.")

    # Use read_img or gray_img as needed
    image3 = (image * 255).astype(np.uint8)

    image2 = readrr.gray_img(image)

    sharpness = features.sharpness(image)
    noise = calculate_image_noise(image)
    exposure_color = calculate_exposure_color(image)
    fluency_feature = calculate_fluency_feature(image)

    # Display metrics in parallel columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Fluency", value=f"{fluency_feature:.2f}")

    with col2:
        st.metric(label="Color Exposure", value=f"{exposure_color:.2f}")

    with col3:
        st.metric(label="Noise", value=f"{noise:.2f}")

    with col4:
        st.metric(label="Sharpness", value=f"{sharpness:.2f}")



    st.markdown("<h2 style='text-align: center; color: white;'> Mirror Line Graph </h2>", unsafe_allow_html=True)

    fig = bilateral_syemmetry(image_path, "With Mirror Line")
    st.pyplot(fig)
    st.text("")
    st.markdown("<h2 style='text-align: center; color: white;'> Local Phase Quantization </h2>", unsafe_allow_html=True)

    mylpq = features.lpq(image2, mode='h')
    st.text("")
    st.markdown("<h2 style='text-align: center; color: white;'> Fractrol Dimension </h2>", unsafe_allow_html=True)

    fd =features.fractal_dimension(image, plot=True)
    st.text("")
    st.markdown("<h2 style='text-align: center; color: white;'> Scale-Invariant Feature Transform (SIFT) </h2>", unsafe_allow_html=True)

    visualize_keypoints(image3)
    st.text("")
    st.write("Texture Analysis:", features.texture_analysis(image2))

