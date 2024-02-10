import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image

# Load your trained model
model = load_model('ResNet50_cifar10_best.h5')

# Define class names for CIFAR-10 labels
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Define a function to make predictions
def predict_image(img):
    # Convert the NumPy array to a PIL Image
    img = Image.fromarray((img * 255).astype(np.uint8))

    # Resize the image to the expected size (32, 32)
    img = img.resize((32, 32))

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    label_index = int(np.argmax(predictions))  # Convert to Python int

    # Get the corresponding class name
    label = class_names[label_index]

    # Get the confidence value for the predicted label
    confidence = float(predictions[0][label_index])

    return {label: confidence}


# Create Gradio interface
io = gr.Interface(fn=predict_image, inputs='image', outputs='label', analytics_enabled=True,
                   title="CIFAR-10 Object Recognition App",
                   description="Made By: Humza Ali [www.linkedin.com/in/humza-ali-se/]")
io.launch()
