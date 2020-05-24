from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import cv2
from net import eval
from tkinter.messagebox import showerror

root = Tk()
root.title('Cat Dog Classifier')

def load_b():
    global img
    global loc
    root.filename = filedialog.askopenfilename(initialdir="D:/deepayan/study/coding/Kaggle", title="Select a File",filetypes=(("jpg files","*.jpg"),("all files","*.*")))
    img = ImageTk.PhotoImage(Image.open(root.filename).resize((128,128),Image.ANTIALIAS))
    loc = root.filename
    img_label = Label(fr1, image=img).place(x=0,y=100)
    
def classify_b():
    global img
    global loc
    x = cv2.resize(cv2.imread(loc),(64,64))
    result = eval(x)
    if result>0.5:
        txt="Dog"
    else:
        txt="Cat"
    lab = Label(fr1, text="It is a " + txt)
    lab.place(x=40,y=240)
    
def report_callback_exception(self, exc, val, tb):
    print(exc)
    print(val)
    print(tb)
    showerror("Error", message="Please upload an Image")

Tk.report_callback_exception = report_callback_exception

fr1 = LabelFrame(root, padx=10, pady=10, width=160,height=280)
fr1.grid_propagate(0)
b_classify = Button(fr1, text="classify", padx=2, pady=2, command=classify_b, width=10)
b_load = Button(fr1, text="load image", padx=2, pady=2, command=load_b, width=10)

fr1.pack(padx=10, pady=10)
b_classify.place(x=30,y=40)
b_load.place(x=30,y=0)

root.mainloop()