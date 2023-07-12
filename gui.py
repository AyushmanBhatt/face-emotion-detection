import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import imutils
import time
import f_Face_info

class FaceInfoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Information")
        self.root.geometry("800x600")
        self.root.resizable(0, 0)

        self.image_path = ""

        # Load background image
        background_image = Image.open("background.jpg")
        background_image = background_image.resize((800, 600), Image.BILINEAR)
        self.background_photo = ImageTk.PhotoImage(background_image)
        self.background_label = tk.Label(self.root, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.canvas = tk.Canvas(self.root, width=600, height=400)
        self.canvas.place(x=100, y=100)

        # Browse button with effects
        self.btn_browse = tk.Button(self.root, text="Browse", command=self.browse_image,
                                    font=("Helvetica", 12, "bold"), bg="#2ecc71", fg="white",
                                    activebackground="#27ae60", activeforeground="white",
                                    relief=tk.RAISED, bd=2)
        self.btn_browse.place(x=350, y=550)

        # Process button with effects
        self.btn_process = tk.Button(self.root, text="Process", command=self.process_image,
                                     font=("Helvetica", 12, "bold"), bg="#3498db", fg="white",
                                     activebackground="#2980b9", activeforeground="white",
                                     relief=tk.RAISED, bd=2)
        self.btn_process.place(x=450, y=550)

        # Webcam button with effects
        self.btn_webcam = tk.Button(self.root, text="Webcam", command=self.open_webcam,
                                    font=("Helvetica", 12, "bold"), bg="#e67e22", fg="white",
                                    activebackground="#d35400", activeforeground="white",
                                    relief=tk.RAISED, bd=2)
        self.btn_webcam.place(x=250, y=550)

        self.video_capture = None
        self.is_webcam_open = False

    def browse_image(self):
        self.image_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                                     filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
        if self.image_path:
            self.display_image()

    def display_image(self):
        image = Image.open(self.image_path)
        image = image.resize((600, 400))
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def process_image(self):
        if self.image_path:
            frame = cv2.imread(self.image_path)
            frame = imutils.resize(frame, width=720)

            start_time = time.time()
            out = f_Face_info.get_face_info(frame)
            end_time = time.time() - start_time
            fps = 1 / end_time

            self.display_result(frame, out, fps)

    def display_result(self, frame, out, fps):
        for data_face in out:
            box = data_face["bbx_frontal_face"]
            if len(box) == 0:
                continue
            else:
                x0, y0, x1, y1 = box
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                step = 13

                try:
                    cv2.putText(frame, "Age: " + data_face["age"], (x0, y0 - 7), font, font_scale, (0, 255, 0), thickness)
                except:
                    pass
                try:
                    cv2.putText(frame, "Gender: " + data_face["gender"], (x0, y0 - step - 10 * 1), font, font_scale, (0, 255, 0), thickness)
                except:
                    pass
                try:
                    cv2.putText(frame, "Race: " + data_face["race"], (x0, y0 - step - 10 * 2), font, font_scale, (0, 255, 0), thickness)
                except:
                    pass
                try:
                    cv2.putText(frame, "Emotion: " + data_face["emotion"], (x0, y0 - step - 10 * 3), font, font_scale, (0, 255, 0), thickness)
                except:
                    pass
                try:
                    cv2.putText(frame, "Name: " + data_face["name"], (x0, y0 - step - 10 * 4), font, font_scale, (0, 255, 0), thickness)
                except:
                    pass

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = image.resize((600, 400))
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        fps_label = tk.Label(self.root, text="FPS: " + str(round(fps, 2)))
        fps_label.place(x=100, y=550)

    def open_webcam(self):
        if not self.is_webcam_open:
            self.video_capture = cv2.VideoCapture(0)
            self.is_webcam_open = True
            self.btn_webcam.config(text="Close Webcam")
            self.process_webcam()
        else:
            self.is_webcam_open = False
            self.btn_webcam.config(text="Webcam")
            self.video_capture.release()

    def process_webcam(self):
        if self.is_webcam_open:
            ret, frame = self.video_capture.read()
            frame = imutils.resize(frame, width=720)

            start_time = time.time()
            out = f_Face_info.get_face_info(frame)
            end_time = time.time() - start_time
            fps = 1 / end_time

            self.display_result(frame, out, fps)

            self.root.after(1, self.process_webcam)
        else:
            self.canvas.delete("all")
            self.canvas.create_rectangle(0, 0, 600, 400, fill="black")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceInfoGUI(root)
    root.mainloop()
