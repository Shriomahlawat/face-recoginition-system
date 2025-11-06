import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Paths
DATASET_DIR = "Dataset"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Load the face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
(width, height) = (130, 100)

@st.cache_resource
def train_model():
    """Train the LBPH model from Dataset folder."""
    names = {}
    images, labels = [], []
    label_id = 0

    for subdirs, dirs, files in os.walk(DATASET_DIR):
        for subdir in dirs:
            names[label_id] = subdir
            subjectpath = os.path.join(DATASET_DIR, subdir)
            for filename in os.listdir(subjectpath):
                path = os.path.join(subjectpath, filename)
                img = cv2.imread(path, 0)
                if img is None:
                    continue
                images.append(img)
                labels.append(label_id)
            label_id += 1

    images = np.array(images)
    labels = np.array(labels)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)
    return model, names

def recognize_face(uploaded_image, model, names):
    """Recognize a face from uploaded image."""
    image = np.array(uploaded_image.convert('RGB'))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected."

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (width, height))
        prediction = model.predict(face_resized)
        if prediction[1] < 800:
            return f"Recognized: {names[prediction[0]]}"
        else:
            return "Unknown face."

    return "No recognizable faces."

def main():
    st.title("🧠 Face Recognition App")
    st.write("Upload an image to identify the person using ML classifier.")

    model, names = train_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")
        result = recognize_face(image, model, names)
        st.success(result)

if __name__ == "__main__":
    main()
