from PIL import Image
import numpy as np
import tensorflow
import streamlit as st

# Load pre-trained model
model = tensorflow.keras.models.load_model("D:\Cat_Dog_Classification\MobileNet.h5", compile=False)
labels = {0: 'Cat', 1: 'Dog'}
st.title("Cat Dog Classification")

# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Upload image through Streamlit
uploaded_image = st.file_uploader("Choose a dog or cat image")

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    img_array = preprocess_image(uploaded_image)
    predictions = model.predict(img_array)

    # Display result
    if predictions[0][0] > predictions[0][1]:
        st.write("Prediction: Dog")
    else:
        st.write("Prediction: Cat")