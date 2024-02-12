import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load trained model
model = load_model("ResNet50_cifar10_best_fr.h5")

# Define the preprocessing function
def preprocess_image(img):
    img = tf.image.resize(img, (32, 32))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img


# Define the postprocessing function
def process_prediction(prediction):
    predicted_class_index = int(prediction.argmax())
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name


# Define the prediction function
def predict_cifar10(img):
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)
    return process_prediction(prediction)

# Create Gradio interface
iface = gr.Interface(
    fn=predict_cifar10,
    inputs=[gr.Image(label="Input Image")],
    outputs=[gr.Label(label="Predicted Class")],
    title="CIFAR-10 Image Classifier",
    description="Upload an image to classify it using a CIFAR-10 model."
)

# Launch the interface
iface.launch()
