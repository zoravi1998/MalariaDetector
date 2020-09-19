from tkinter import *
from PIL import Image,ImageTk
from glob import glob
import numpy as np
from termcolor import colored
test=[]
title=[]
test_img=[]

root = Toplevel()
root.geometry('360x240')
root.title('Blood Samples Result')


my_label=0
button_forward=0
button_back=0
text_pred=0
l=0

def forward(image_number):
    global my_label
    global button_forward
    global button_back
    global text_pred
    global l
    my_label.grid_forget()
    
    my_label = Label(root,image=test_img[image_number-1])
    text_pred=Label(root,text=title[image_number][0],font='Times 16 bold',fg=title[image_number][1])
    text_pred.grid(row=1,column=2)
    button_forward=Button(root,text='>>',command=lambda:forward(image_number+1))
    button_back=Button(root,text='<<',padx=0,command=lambda:backward(image_number-1))
    if(image_number==l-1):
        button_forward=Button(root,text='>>',state=DISABLED)
    my_label.grid(row=1,column=1)
    button_back.place(x=50,y=180)
    button_forward.place(x=200,y=180)

def backward(image_number):
    global my_label
    global button_forward
    global button_back
    global text_pred
    global l
    my_label.grid_forget()
    my_label = Label(root,image=test_img[image_number-1])
    text_pred=Label(root,text=title[image_number][0],font='Times 16 bold',fg=title[image_number][1])
    text_pred.grid(row=1,column=2)
    button_forward=Button(root,text='>>',command=lambda:forward(image_number+1))
    button_back=Button(root,text='<<',padx=0,command=lambda:backward(image_number-1))
    if(image_number==1):
        button_forward=Button(root,text='>>',state=DISABLED)
    my_label.grid(row=1,column=1)
    button_back.place(x=50,y=180)
    button_forward.place(x=200,y=180)
    


def main(test1,title1):
    global test,title,my_label,button_forward,button_back,text_pred,l
    test=test1
    title=title1
    l=test.size
    for i in range (l):
        x=Image.open(test[i])
        #sx.resize((200,200),Image.ANTIALIAS)
        test_img.append(ImageTk.PhotoImage(x))
        
    text_l=Label(root,text='Blood Sample',padx=80)
    text_l.grid(row=0,column=1)
    my_label= Label(root,image=test_img[0])
    my_label.grid(row=1,column=1)
    text_pred=Label(root,text=title[0][0],font='Times 16 bold',fg=title[0][1])
    text_pred.grid(row=1,column=2)
    button_back=Button(root,text='<<',padx=0,command=backward)
    button_forward=Button(root,text='>>',command=lambda:forward(2))
    button_back.place(x=50,y=180)
    button_forward.place(x=200, y=180)
    
    root.mainloop()


        