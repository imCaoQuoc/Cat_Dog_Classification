from PIL import Image
import numpy as np
import tensorflow
import streamlit as st

# Load pre-trained model
model = tensorflow.keras.models.load_model("D:\Cat_Dog_Classification\MobileNet.h5", compile=False)
labels = {0: 'Cat', 1: 'Dog'}
st.title("Cat Dog Classification")

# Function to preprocess the image
def process_image(image_path, pretrained_model):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image)
    predictions = pretrained_model.predict(np.expand_dims(image_array, axis=0))
    predicted_class = np.argmax(predictions)
    confidence_score = predictions[0, predicted_class]

# Upload image through Streamlit
uploaded_image = st.file_uploader("Choose a dog or cat image")

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    img_array = process_image(uploaded_image, model)
    predictions = model.predict(img_array)

    # Display result
    if predictions[0][0] > predictions[0][1]:
        st.write("Prediction: Dog")
    else:
        st.write("Prediction: Cat")