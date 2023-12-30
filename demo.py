import cv2
import numpy as np
import tensorflow
import streamlit as st

model = tensorflow.keras.models.load_model("MobileNet.h5", compile=False)
labels = {0: 'Cat', 1: 'Dog'}
st.title("Cat Dog Classification")