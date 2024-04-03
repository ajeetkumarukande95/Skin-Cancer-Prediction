import gradio as gr
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

def predict(image_file):
    width, height = 224, 224
    img = image.img_to_array(image_file)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    model = load_model("/vgg16_model.h5")
    result = model.predict(img)

    if result[0][0] >= 0.5:
        prediction = "Malignant"
    else:
        prediction = "Benign"

    return prediction

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Textbox(label="Predicted Results"),
    title="Skin Cancer Prediction",
    description="Upload an image containing skin lesion to predict if it is malignant or benign.",
    theme="huggingface",
    allow_flagging=False,
    examples=[
        ["example.jpg"],  # Replace "example.jpg" with actual example image file path
    ]
)

iface.launch(share=True)
