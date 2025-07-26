import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("Digit_classificatoin_MNIST.keras")

# Prediction function
def predict_digit(img):
    if img is None:
        return "❌ No image uploaded!"
    
    img = img.convert("L").resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    return f"🔢 Predicted Digit: {predicted_digit}"

# Example images (can use local paths or leave empty for placeholders)
example_list = [
    ["examples/0.jpeg"],
    ["examples/1.png"],
    ["examples/2.png"],
    ["examples/3.jpeg"],
    ["examples/4.png"],
    ["examples/5.png"],
    ["examples/6.jpeg"],
    ["examples/7.png"],
    ["examples/8.png"],
    ["examples/9.png"]
]

# Gradio Interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", label="🖼️ Upload a Digit (0–9)"),
    outputs=gr.Textbox(label="📢 Prediction"),
    title="🎯 MNIST Digit Classifier",
    description="""
        ✨ Upload a handwritten digit image.<br>
        Our neural network will try to guess what number it is!<br>
        <br>🧠 Built with TensorFlow & Gradio · Try the examples below 👇
    """,
    examples=example_list,
    allow_flagging="never"
)

demo.launch()
