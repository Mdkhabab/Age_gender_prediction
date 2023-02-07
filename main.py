import cv2,ctypes,os
from tkinter import *
import threading as tr
from PIL import ImageTk, Image
import tkinter.messagebox as tkMessageBox
from tensorflow.keras.models import load_model
from tkinter.filedialog import askopenfilename

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model =  load_model('./model/model.h5')
model2 =  load_model('./model/gender.h5')

classes = ['0-2','3-6','7-14','15-24', '25-37', '38-47', '48-59','60-100']

def realtime():
        def pred():
                cap = cv2.VideoCapture(0)
                while 1:
                        ret, img = cap.read()
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        for (x,y,w,h) in faces:
                                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
                                roi_gray = gray[y:y+h, x:x+w]
                                roi_color = img[y:y+h, x:x+w]
                                roi_color2 = roi_color.copy()
                                roi_color = cv2.resize(roi_color,(96,96))
                                roi_color = roi_color.reshape(-1,96,96,3)/255.0
                                pred = model.predict(roi_color)
                                pred = int(pred[1][0])
                                txt = ''
                                for i in classes:
                                        j = i
                                        a,b=i.split('-')
                                        if pred>=int(a) and pred<=int(b):
                                                txt = j
                                                
                                roi_color = cv2.resize(roi_color2,(150,150))
                                roi_color = roi_color.reshape(-1,150,150,3)/255.0
                                pred2 = model2.predict(roi_color)
                                pred2 = float(pred2[0][0])*100
                                if pred2>50:
                                        txt+=' (Female)'
                                else:
                                        txt+=' (Male)'
                                cv2.putText(img,txt, (x+5,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100),2)
                        cv2.imshow('Age and Gender Prediction',img)
                        k = cv2.waitKey(30) & 0xff
                        if k == 27:
                                break
                cap.release()
                cv2.destroyAllWindows()
        tr.Thread(target=pred).start()
        
def Exit():
    global home
    result = tkMessageBox.askquestion(
        "Age and Gender Detection", 'Are you sure you want to exit?', icon="warning")
    if result == 'yes':
        home.destroy()
        exit()
    else:
        tkMessageBox.showinfo(
            'Return', 'You will now return to the main screen')

def browse():
    global file,l1
    try:
        l1.destroy()
    except:
        pass
    file = askopenfilename(initialdir=os.getcwd(), title="Select Image", filetypes=( ("images", ".png"),("images", ".jpg"),("images", ".jpeg")))

def predict():
    global file,l1
    if file!='' or file!= None:
        img = cv2.imread(file)
        img = cv2.resize(img,(640,480))
        imgc=img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
                cv2.rectangle(imgc,(x,y),(x+w,y+h),(255,255,0),2)
                img = img[y:y+h, x:x+w]
                roi_color2=img.copy()
                img = cv2.resize(img,(96,96)) 
                img = img.reshape(-1,96,96,3)/255.0
                pred = model.predict(img)
                pred = int(pred[1][0])
                txt = ''
                for i in classes:
                        j = i
                        a,b=i.split('-')
                        if pred>=int(a) and pred<=int(b):
                                txt = j
                roi_color = cv2.resize(roi_color2,(150,150))
                roi_color = roi_color.reshape(-1,150,150,3)/255.0
                pred2 = model2.predict(roi_color)
                pred2 = float(pred2[0][0])*100
                if pred2>50:
                        txt+=' (Female)'
                else:
                        txt+=' (Male)'
                cv2.putText(imgc,txt, (x-15,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100),2)
        cv2.imshow('Age and Gender Prediction',imgc)
        cv2.waitKey(1)

home = Tk()
home.title("Age and Gender Detection")

img = Image.open("images/home.png")
img = ImageTk.PhotoImage(img)
panel = Label(home, image=img)
panel.pack(side="top", fill="both", expand="yes")
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
lt = [w, h]
a = str(lt[0]//2-450)
b= str(lt[1]//2-320)
home.geometry("900x653+"+a+"+"+b)
home.resizable(0,0)

    
photo = Image.open("images/1.png")
img2 = ImageTk.PhotoImage(photo)
b1=Button(home, highlightthickness = 0, bd = 0,activebackground="#2b4b47", image = img2,command=browse)
b1.place(x=0,y=209)

photo = Image.open("images/2.png")
img3 = ImageTk.PhotoImage(photo)
b2=Button(home, highlightthickness = 0, bd = 0,activebackground="#2b4b47", image = img3,command=predict)
b2.place(x=0,y=282)

photo = Image.open("images/3.png")
img4 = ImageTk.PhotoImage(photo)
b3=Button(home, highlightthickness = 0, bd = 0,activebackground="#2b4b47", image = img4,command=realtime)
b3.place(x=0,y=354)

photo = Image.open("images/4.png")
img5 = ImageTk.PhotoImage(photo)
b4=Button(home, highlightthickness = 0, bd = 0,activebackground="#2b4b47", image = img5,command=Exit)
b4.place(x=0,y=426)

home.mainloop()



