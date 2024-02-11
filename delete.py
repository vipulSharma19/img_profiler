import streamlit as st
import cv2
import numpy as np
from skimage import color, filters
from streamlit_option_menu import option_menu
st.set_page_config(
    page_title="Image Analysis App",
    page_icon=":camera:",
    layout="wide",
)
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Json"],
        icons=["house", "gear", ],
        menu_icon="cast",
        default_index=0,

        # orientation = "horizontal",
    )
# st.set_option('footer',False)
def calculate_sharpness_and_focus(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    focus = cv2.Laplacian(gray, cv2.CV_64F).mean()
    return sharpness, focus

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

# Sample image path
image_path = r"C:\Users\sharm\PycharmProjects\Image_prof\129.png"
img = cv2.imread(image_path)

# Set Streamlit app theme


# Display the image
st.image(img, caption='Uploaded Image', use_column_width=True)

# Calculate and display image analysis metrics
sharpness, focus = calculate_sharpness_and_focus(img)
noise = calculate_image_noise(img)
exposure_color = calculate_exposure_color(img)
fluency_feature = calculate_fluency_feature(img)

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

st.write(f"Focus: {focus:.2f}")
# Remove Streamlit footer
custom_css = """
<style>
footer {
    display: none;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)