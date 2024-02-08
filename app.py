import gradio as gr
from tensorflow.keras.models import load_model

model = load_model('model.h5') # Remember to change model name to correct model name
