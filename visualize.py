import streamlit as st
from PIL import Image
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from streamlit_image_comparison import image_comparison
import time

st.title('Super Resolution Models')


st.sidebar.title('Super Resolution Models')
selected_model = st.sidebar.selectbox('Select Model', ('RRDB_PSNR_x4.pth', 'RRDB_ESRGAN_x4.pth', 'GFPGANv1.4.pth'))

st.sidebar.title('Input Image')
input_image = st.sidebar.file_uploader("Choose an input image", type=['png', 'jpg', 'jpeg'])
if input_image:
    Image.open(input_image).save("input_image.png")
    image = Image.open(input_image)
    height, width = image.size
    st.sidebar.markdown('**Input Image Dimensions**')
    st.sidebar.write('Width:', width)
    st.sidebar.write('Height:', height)
    st.sidebar.image(input_image, caption='Input Image')


model_path = f"models/{selected_model}"
device = torch.device('cpu')
model = None
model_loaded = False

with st.spinner('Loading Model...'):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    model_loaded = True
    st.success('Model Loaded Successfully')


st.header(f'Selected Model: {selected_model}', divider='rainbow', anchor='center')


if model_loaded:
    start_model = st.button('Start Super Resolution')

    if start_model:
        if not input_image:
            st.error('Please Upload an Image')
        else:
            with st.spinner('Processing Image...'):
                img_path = "input_image.png"
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = img * 1.0 / 255
                img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
                img_LR = img.unsqueeze(0)
                img_LR = img_LR.to(device)

                with torch.no_grad():
                    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round()
                cv2.imwrite('output.png', output)

            st.success('Super Resolution Complete!')
            st.write('Download Super Resolved Image:')
            st.download_button('Download', 'output.png', key='download_button')


if st.button("Show Image Comparison"):
    if input_image and model_loaded:
        image_comparison(
            img1="input_image.png",
            img2="output.png",
        )
    else:
        st.warning("Please upload an image and load a model to compare.")

