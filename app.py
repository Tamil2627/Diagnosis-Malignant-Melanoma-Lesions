import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

# Load the pre-trained Keras model
model = load_model("models/model_80_classes.keras")

# Dictionary mapping class indices to class names
classes = {
    0: 'benign',
    1: 'malignant',
    # Add more class names as needed
}

# Initialize GUI
top = tk.Tk()
top.geometry('1200x600')
top.title("Diagnosis Malignant Melanoma Lesions")
top.configure(background='#8e44ad')

prediction_label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'), pady=10)
sign_image = Label(top)

def classify(file_path):
    try:
        image = Image.open(file_path)
        image = image.resize((299, 299))  # Resize to match the model's input shape
        image = np.array(image)
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        pred = model.predict(image)
        pred_class = np.argmax(pred)
        sign = classes[pred_class]
        prediction_label.configure(foreground='#011638', text=f"Predicted Class: {sign}")
    except Exception as e:
        print(f"Error: {e}")
        print("Image classification failed.")

def show_classify_button(file_path):
    try:
        classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
        classify_b.configure(background='#364198', foreground='white', font=('arial', 10, 'bold'))
        classify_b.place(relx=0.79, rely=0.46)
    except:
        print("Failed to create classify button")

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        prediction_label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to upload image")

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=20)

sign_image.pack(side=TOP, expand=True)
prediction_label.pack(side=TOP, expand=True)

heading = Label(top, text="Diagnosis Malignant Melanoma Lesions", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()
