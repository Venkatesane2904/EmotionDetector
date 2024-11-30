import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model_path = "E:\GUVI\Project\EmotionDetector\model.h5"  # Ensure this file is in the same directory as this script
model = load_model(model_path)

# Define emotion classes
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(uploaded_image):
    """
    Preprocess the uploaded image for emotion detection:
    - Convert to grayscale
    - Resize to 48x48
    - Convert grayscale to RGB (3 channels)
    - Normalize the pixel values
    """
    # Convert the uploaded image to an OpenCV-compatible format (numpy array)
    image = np.array(uploaded_image)

    # Check if the image is a single-channel (grayscale) or multi-channel (color)
    if len(image.shape) != 3:
        # If the image is grayscale, expand dimensions to (height, width, 1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 48x48
    resized_image = cv2.resize(gray_image, (48, 48))
    
    # Convert grayscale to RGB (stack grayscale into 3 channels)
    rgb_image = cv2.merge([resized_image, resized_image, resized_image])
    
    # Normalize the pixel values to the range [0, 1]
    normalized_image = rgb_image / 255.0
    
    # Expand dimensions to match model input shape (1, 48, 48, 3)
    input_image = np.expand_dims(normalized_image, axis=0)
    
    return input_image

def predict_emotion(image):
    """
    Predict the emotion from the uploaded image.
    """
    input_image = preprocess_image(image)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_emotion = emotion_classes[predicted_class_index]
    return predicted_emotion, predictions

# Streamlit application
st.title("Emotion Detection App")
st.write("Upload an image to detect the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Process and predict emotion
    try:
        emotion, probabilities = predict_emotion(image)
        st.write(f"**Detected Emotion:** {emotion}")
        st.write(f"**Class Probabilities:** {probabilities}")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
