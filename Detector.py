import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler
import os
import os.path
from tkinter import *
from PIL import Image,ImageTk
from glob import glob
import numpy as np
from termcolor import colored
from tkinter import filedialog

os.environ['TORCH_HOME']=os.getcwd()
model =models.resnet50(pretrained=True)

#Directory Browser
app=Tk()
app.geometry("300x260")
app.title('Malaria Detector App')
f_name=Entry(app,width=30)
test=[]
title=[]
l=0

def browse():
    #global my_image
    global f_name
    app.filename = filedialog.askdirectory(initialdir="D:/",title="Select A Folder")
    f_name.insert(0,app.filename)
def open():
    global f_name
    global test
    test=np.array(glob(f_name.get()+'/*'))
    #my_label = Label(app,text=app.filename).pack()
    print(test)
    #img=Image.open(test[1])
    #plt.imshow(img)
    #plt.show()
f_name.place(x=30,y=40)
button_bro=Button(app,text="Browse",command=browse).place(x=40,y=70)
button_opn=Button(app,text="Open",command=open).place(x=160,y=70)





def initialize():
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, 2, bias=True)

    fc_parameters = model.fc.parameters()

    for param in fc_parameters:
        param.requires_grad = True
    model.load_state_dict(torch.load('malaria_detection.pt', map_location=lambda storage, loc: storage))

def load_input_image(img_path):    
    image = Image.open(img_path)
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    return image

def predict_malaria(model, class_names, img_path):
    # load the image and return the predicted breed
    img = load_input_image(img_path)
    model = model.cpu()
    model.eval()
    o=model(img)
    #output = torch.exp(o)
    #probs, classes = output.topk(1, dim=1)
    #print(probs.item())
    #print(o)
    idx = torch.argmax(o)
    return class_names[idx]


def run():
    global l
    initialize()
    class_names=['Parasitized','Uninfected']
    l=test.size
    for i in range(l):
        img_path=test[i]
        if predict_malaria(model, class_names, img_path) == 'Parasitized':
            title.append(['Parasitized','red'])
        else:
            title.append(['Unifected','green'])
    load_label=Label(app,text="Total Samples Analyzed : "+str(l),font="Calibri 12 bold")
    load_label.grid(row=4,column=1)
    print(title)
    
def view_res():
    import imageviewer1
    imageviewer1.main(test,title)
    
button_run = Button(app,text="Run",command=run).place(x=100,y=100)
button_view=Button(app,text="View Result",command=view_res).place(x=80,y=140)

def main():
 app.mainloop()
 initialize()
if __name__=="__main__":
 main()
