import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import io

# Load the trained model
model = load_model('mnist_v4_99.48.h5')

# Initialize variables for drawing
line_width = 10
canvas_width = 280
canvas_height = 280
drawing = False

# Create a drawing area
def start_drawing(event):
    global drawing
    drawing = True

def draw(event):
    global drawing
    if drawing:
        x, y = event.x, event.y
        canvas.create_oval(x - line_width, y - line_width, x + line_width, y + line_width, fill='black', outline='black')

def stop_drawing(event):
    global drawing
    drawing = False

# Function to recognize the digit
def recognize_digit():
    global canvas
    
    # Convert the canvas content to an image
    canvas_image = Image.new('L', (canvas_width, canvas_height), 'white') # 'L' indicates that it's a grayscale image.
    ps_data = canvas.postscript(colormode='color')
    img = Image.open(io.BytesIO(ps_data.encode('utf-8')))
    canvas_image.paste(img, (10, 10))
    
    # Resize and normalize the image
    img = canvas_image.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict the digit using the loaded model
    prediction = model.predict(img_array)
    
    # Get the predicted digit (the class with the highest probability)
    digit = np.argmax(prediction)
    
    # Update the label with the recognized digit
    result_label.config(text=f"Recognized Digit: {digit} : {prediction}")

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    result_label.config(text="")  # Set the text to an empty string

# Create the main window
root = tk.Tk()
root.title("Digit Recognition")

# Create a canvas for drawing
canvas = Canvas(root, bg='white', width=canvas_width, height=canvas_height)
canvas.pack()

# Create a Recognize button
recognize_button = Button(root, text="Recognize Digit", command=recognize_digit)
recognize_button.pack()

# Create a Clear Screen button
clear_button = Button(root, text="Clear Screen", command=clear_canvas)
clear_button.pack(expand = True)

# Create a label to display the recognized digit
result_label = Label(root, text="", font=("Helvetica", 12))
result_label.pack()


# Bind mouse events to the canvas
canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)

# Start the Tkinter main loop
root.mainloop()
