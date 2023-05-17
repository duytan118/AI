import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from tensorflow import keras

class SkinCancerTesterGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Skin Cancer Prediction")
        self.width = 800  # Chiều rộng của cửa sổ
        self.height = 700  # Chiều cao của cửa sổ
        # Lấy kích thước màn hình
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Tính toán vị trí của cửa sổ
        x = int((screen_width - self.width) / 2)
        y = int((screen_height - self.height) / 2)

        # Đặt vị trí của cửa sổ
        self.window.geometry(f"{self.width}x{self.height}+{x}+{y}")
        self.window.configure(bg='beige')
        
        self.screen = tk.Label(self.window)
        self.screen.place(x=0, y=0, width=800, height=700)
        
        self.load_image = tk.Label(self.window, relief=tk.SUNKEN, bd=3)
        self.load_image.place(x=250, y=60, width=300, height=300)
        self.load_image.configure(relief='solid', borderwidth=2)
        
        
        self.browse_button = tk.Button(self.window, text="BROWSER", font=("Arial", 14), command=self.browse_image)
        self.browse_button.place(x=310, y=600, width=160, height=40)
        self.browse_button.configure(bg='blue', fg='white')
        
        
        self.label_2 = tk.Label(self.window, relief=tk.SUNKEN, bd=3, font=("Arial", 12), text="RESULT")
        self.label_2.place(x=120, y=500, width=550, height=40)
        
        self.diagnostic_button = tk.Button(self.window, text="PREDICT", font=("Arial", 18), command=self.predict_image)
        self.diagnostic_button.place(x=290, y=400, width=200, height=50)
        self.diagnostic_button.configure(bg='green', fg='white')

        # Load model
        self.model = keras.models.load_model('D:\AI\FINAL.h5')  # Update with your actual model path
        self.class_labels = {0: "UNG THƯ BIỂU MÔ TẾ BÀO VẢY(SQUAMOUS CELL CARCINOMA)", 1: "UNG THƯ BIỂU MÔ TẾ BÀO ĐÁY(BASAL CELL CARCINOMA)", 2: "UNG THƯ CÁC TUYẾN PHỤ THUỘC DA(ADNEXAL GLAND TUMORS)"}

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            img = Image.open(file_path)
            img = img.resize((300, 300))
            img = ImageTk.PhotoImage(img)
            self.load_image.configure(image=img)
            self.load_image.image = img
            self.predict_image(file_path)

    def predict_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((150, 150))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)
        class_index = np.argmax(prediction)
        class_label = self.class_labels[class_index]
        self.label_2.config(text=class_label)

    def perform_diagnostic(self):
        file_path = self.line_edit.get()
        if file_path:
            self.predict_image(file_path)
        
        
if __name__ == "__main__":
    app = SkinCancerTesterGUI()
    app.window.mainloop()
