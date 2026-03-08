import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model.keras")

def predict_image(img):
    # img is already a PIL Image object because type="pil"
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)   # Correct: raw images (0-255) are expected by the model's preprocess_input layer
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)

    print("Raw prediction:", prediction)

    return "Cat 🐱" if class_id == 0 else "Dog 🐶"

ui = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(label="Prediction"),
    title="Dog vs Cat Image Classifier",
    description="By Ashika Thakur"
)

ui.launch()