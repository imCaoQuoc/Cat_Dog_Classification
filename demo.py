from PIL import Image
import numpy as np
import tensorflow
import streamlit as st

# Load pre-trained model
model = tensorflow.keras.models.load_model("MobileNet.h5", compile=False)
labels = {0: 'Cat', 1: 'Dog'}
st.title("Cat Dog Classification")

# Function to preprocess the image
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    predictions = model.predict(np.expand_dims(image_array, axis=0))
    predicted_class = np.argmax(predictions)
    confidence_score = predictions[0, predicted_class]

    #display result
    if predicted_class == 0:
        st.write("Cat")
    elif predicted_class == 1: 
        st.write("Dog")
    st.write("Confidence Score:", confidence_score)

# Upload image through Streamlit
uploaded_image = st.file_uploader("Choose a dog or cat image")

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    # Preprocess and predict
    img_array = process_image(uploaded_image)
