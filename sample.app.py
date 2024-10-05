import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
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
top.configure(background='#FFFFFF')

# Function to classify the lesion
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
        messagebox.showerror("Error", "Image classification failed.")
        print(f"Error: {e}")

# Function to handle image upload
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(uploaded)
            sign_image.configure(image=im)
            sign_image.image = im
            prediction_label.configure(text='')
            show_classify_button(file_path)
    except Exception as e:
        messagebox.showerror("Error", "Failed to upload image.")
        print(f"Error: {e}")

# Function to display the classify button after image upload
def show_classify_button(file_path):
    try:
        classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
        classify_b.configure(background='#364198', foreground='white', font=('arial', 10, 'bold'))
        classify_b.place(relx=0.79, rely=0.46)
    except Exception as e:
        messagebox.showerror("Error", "Failed to create classify button.")
        print(f"Error: {e}")

# Button for uploading an image
upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=20)

# Label to display the uploaded image
sign_image = Label(top)
sign_image.pack(side='top', expand=True)

# Label to display the classification result
prediction_label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'), pady=10)
prediction_label.pack(side='top', expand=True)

# Label to display the heading
heading = Label(top, text="Diagnosis Malignant Melanoma Lesions", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()
