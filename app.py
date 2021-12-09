
import pandas as pd
import matplotlib as plt
import os
import math
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from math import *
from PIL import Image

# load model et data
model = load_model('model.h5')
data_test = pd.read_csv("data/test.csv")

# header

header_img = Image.open(r"img\title.jpg")
st.image(header_img)

st.markdown("Bienvenue sur notre app web de reconnaissance de chiffres pour data scientist \o/")

#mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=1000,
    height=150,
    drawing_mode="freedraw",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)

if st.button('Envoi VOYANCE au 8 12 12'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(1, 28, 28, 1))
    st.write(f'Notre prédiction: {np.argmax(val[0])}')
