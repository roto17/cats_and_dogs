import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

st.set_page_config(page_title='Cats vs Dogs Classifier', page_icon='🐾')

@st.cache_resource
def load_clf():
    return load_model('best_model_cats_dogs.keras')

model = load_clf()

st.title('🐾 Cats vs Dogs Classifier')
st.write('Chargez une image pour obtenir la prédiction du modèle.')

uploaded = st.file_uploader('Choisir une image...', type=['jpg', 'jpeg', 'png'])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Image chargée', use_column_width=True)

    arr = img_to_array(img.resize((150, 150))) / 255.0
    arr = np.expand_dims(arr, axis=0)
    proba = model.predict(arr)[0][0]

    label = 'Dog 🐶' if proba >= 0.5 else 'Cat 🐱'
    confidence = proba if proba >= 0.5 else 1 - proba

    st.subheader(f'Prédiction: **{label}**')
    st.write(f'Confiance: {confidence*100:.1f}%')
    st.progress(float(confidence))