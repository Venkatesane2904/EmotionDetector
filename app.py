
"""import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
# model = load_model('emotion_model.h5')

model = load_model(r"E:\GUVI\Project\EmotionDetector\model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Emotion Detection from Images")
st.write("Upload an image to detect emotions.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.write("No faces detected in the image. Please try another image.")
    else:
        for (x, y, w, h) in faces:
            face = gray_image[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Predict emotion
            predictions = model.predict(face)[0]
            emotion = emotion_labels[np.argmax(predictions)]

            # Draw bounding box and emotion on image
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the results
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
        """
        
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
