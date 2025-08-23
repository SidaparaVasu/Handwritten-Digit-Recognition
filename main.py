import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageOps, ImageGrab
import numpy as np
from keras.models import load_model
import shutil
import io

# -------------------------------
# Utility Functions
# -------------------------------

def is_ghostscript_installed():
    """
    Check if Ghostscript is available in system PATH.
    """
    return shutil.which("gs") or shutil.which("gswin64c") or shutil.which("gswin32c")


def capture_canvas(canvas, root):
    """
    Capture the canvas content as a grayscale PIL image.
    Chooses method based on Ghostscript availability.
    """
    if use_ghostscript:
        # Method using canvas.postscript
        ps_data = canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps_data.encode('utf-8')))
        img = img.convert('L')
    else:
        # Method using ImageGrab
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
    return img


def preprocess_image(img):
    """
    Resize to 28x28, invert colors, normalize, and reshape for model input.
    """
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 28, 28, 1)


# -------------------------------
# Application Functions
# -------------------------------

def start_drawing(event):
    global drawing
    drawing = True

def draw(event):
    if drawing:
        x, y = event.x, event.y
        canvas.create_oval(
            x - line_width, y - line_width,
            x + line_width, y + line_width,
            fill='black', outline='black'
        )

def stop_drawing(event):
    global drawing
    drawing = False

def recognize_digit():
    """
    Capture canvas, preprocess, predict using model, and display result.
    """
    img = capture_canvas(canvas, root)
    img_array = preprocess_image(img)

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    result_label.config(text=f"Recognized Digit: {digit} : {np.round(prediction[0], 3)}")

def clear_canvas():
    canvas.delete("all")
    result_label.config(text="")


# -------------------------------
# Main Program
# -------------------------------

# Load model
model = load_model('./Models/mnist_v1_99.51.h5')

# Check Ghostscript availability
use_ghostscript = is_ghostscript_installed()
if use_ghostscript:
    print("Ghostscript detected: using PostScript method.")
else:
    print("Ghostscript not detected: using ImageGrab method.")

# Initialize drawing variables
line_width = 10
canvas_width = 280
canvas_height = 280
drawing = False

# Create main window
root = tk.Tk()
root.title("Digit Recognition")

# Create Canvas
canvas = Canvas(root, bg='white', width=canvas_width, height=canvas_height)
canvas.pack(pady=10)

# Buttons
recognize_button = Button(root, text="Recognize Digit", command=recognize_digit)
recognize_button.pack(pady=5)

clear_button = Button(root, text="Clear Screen", command=clear_canvas)
clear_button.pack(pady=5)

# Label to display results
result_label = Label(root, text="", font=("Helvetica", 12))
result_label.pack(pady=5)

# Bind drawing events
canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)

# Start Tkinter main loop
root.mainloop()
