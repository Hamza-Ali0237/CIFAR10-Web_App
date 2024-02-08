import gradio as gr
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

model = load_model('model.h5') # Remember to change model name to correct model name

# Define a function to make predictions
def predict_image(img):
    # Preprocess the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make predictions
    predictions = model.predict(img)
    label = np.argmax(predictions)

    return label

# Create Gradio interface
io = gr.Interface(fn=predict_image, inputs='image', outputs='label', analytics_enabled=True,
	title="CIFAR-10 Object Recognition App", description="Made By: Humza Ali [www.linkedin.com/in/humza-ali-se/]")
io.launch()