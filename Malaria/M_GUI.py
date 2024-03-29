#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf  # Replace with your actual model import

# Load your trained machine learning model here
# Replace 'your_model_path' with the actual path to your model file
model = tf.keras.models.load_model('malaria_detection_model_rf.h5')

# Function to classify an image and display it
def classify_image():
    file_path = filedialog.askopenfilename()  # Open a file dialog to choose an image
    if file_path:
        # Load and preprocess the image for inference (you'll need to adapt this part)
        image = Image.open(file_path)
        image = image.resize((64, 64))  # Resize the image to match your model's input size
        image = np.array(image) / 255.0  # Normalize the image data
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Perform inference with your model
        prediction = model.predict(image)

        # Determine the class (Infected or Uninfected)
        if prediction[0][0] > 0.5:
            result_label.config(text="Infected", fg="red")
        else:
            result_label.config(text="Uninfected", fg="green")

        # Display the image in the GUI
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Create the main application window
app = tk.Tk()
app.title("Malaria Infection Classifier")

# Create labels, buttons, and image display area with decorations
title_label = tk.Label(
    app,
    text="Malaria Infection Classifier",
    font=("Helvetica", 20, "bold"),
    bg="blue",
    fg="white"
)
title_label.pack(pady=10, fill=tk.X)

classify_button = tk.Button(
    app,
    text="Classify Image",
    command=classify_image,
    font=("Helvetica", 14),
    bg="green",
    fg="black"
)
classify_button.pack(pady=10)

result_label = tk.Label(
    app,
    text="",
    font=("Helvetica", 16),
    fg="black"
)
result_label.pack(pady=10)

image_label = tk.Label(app)
image_label.pack()

# Start the Tkinter main loop
app.mainloop()


# In[ ]:




