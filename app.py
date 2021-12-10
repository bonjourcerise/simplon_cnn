# import
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st


# load model and data
model = tf.keras.models.load_model('model.h5')
data = pd.read_csv("data/test.csv")

# reshape input
def reshape(input):
    reshape_input = input / 255
    reshape_input = np.array(input).reshape((-1, 28, 28, 1))
    return reshape_input

# view img
def view_img(input):
    img = np.array(input).reshape([28,28])
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.get_cmap('gray'))
    st.pyplot(fig)

# prediction
def pred():
    mysample = data.sample(n=1)
    view_img(mysample)
    prediction = np.argmax(model.predict(reshape(mysample)), axis=1)
    return str(prediction)[1]

# sidebar
st.sidebar.markdown("### App CNN / MNIST")
st.sidebar.markdown("Bienvenue sur mon app web de reconnaissance de chiffres pour data scientist \o/")

# content
st.markdown('#')
col1, col2 = st.columns(2)

with col1:
    st.write("**Prédiction** : " + pred())

with col2:
    value = st.selectbox("Est-ce que la prédiction est ok ?", ["Oui, ce modèle est incroyable", "Non, ce modèle est nul"])

    if st.button("Valider la réponse"):
        if value == "Oui, ce modèle est incroyable":
            st.write("Merci, je sais")
        elif value != "Non, ce modèle est nul":
            st.write("Alors recommence")