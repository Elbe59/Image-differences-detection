import os
from tkinter import *
from PIL import ImageTk, Image
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

root = Tk()
root.title('RESULT_IMG_6567.JPG')
#root.geometry("1200x800")

my_img1 = ImageTk.PhotoImage(Image.open("output/Chambre/RESULT_IMG_6567.JPG").resize((600,400), Image.ANTIALIAS))
my_img2 = ImageTk.PhotoImage(Image.open("output/Chambre/RESULT_IMG_6569.JPG").resize((600,400), Image.ANTIALIAS))
my_img3 = ImageTk.PhotoImage(Image.open("output/Chambre/RESULT_IMG_6568.JPG").resize((600,400), Image.ANTIALIAS))
my_img4 = ImageTk.PhotoImage(Image.open("output/Chambre/RESULT_IMG_6567.JPG").resize((600,400), Image.ANTIALIAS))
my_img5 = ImageTk.PhotoImage(Image.open("output/Chambre/RESULT_IMG_6570.JPG").resize((600,400), Image.ANTIALIAS))
image_list = [my_img1, my_img2, my_img3, my_img4, my_img5]
original_img = Label(image=my_img1)
result_img = Label(image=my_img2)
original_img.grid(row=1, column=0,ipady=5)
result_img.grid(row=1, column=1,ipady=5)

# Images labels
label_original = Label(root,text="Original Image")
label_original.grid(row=0,column=0,ipady=5)
label_result = Label(root,text="Result Image")
label_result.grid(row=0, column=1,ipady=5)

# labelText1.pack()

#

def add_image(image):
    my_img = ImageTk.PhotoImage(image.resize((600, 400), Image.ANTIALIAS))
    #im


def show_image():
    pass

def flatten(t):
    return [item for sublist in t for item in sublist]

def show_confusion_matrix():
    #https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/
    fig = plt.figure(figsize=(4,4),dpi=60)
    cf_matrix = [[73,7],[7,141]]

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    flatten(cf_matrix)]

    group_percentages = ["{0:.2%}".format(value) for value in
                         flatten(cf_matrix) / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Confusion Matrix of this image')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    if not os.path.exists('./temp/'):
        os.makedirs('./temp/')
    plt.savefig('./temp/confusion_matrix.png')
    img = ImageTk.PhotoImage(Image.open("./temp/confusion_matrix.png"))
    result_img = Label(image=img)
    result_img.image = img  # !!! It is necessary to keep a reference of the image, otherwise the image won't display
    result_img.grid(row=2, column=1,ipady=5)
    ## Display the visualization of the Confusion Matrix.
    #plt.show()



def forward(image_number):
    global result_img
    global button_forward
    global button_back

    result_img.grid_forget()
    root.title('RESULT_IMG_6568.JPG')
    result_img = Label(image=image_list[image_number - 1])
    button_forward = Button(root, text=">>", command=lambda: forward(image_number + 1))
    button_back = Button(root, text="<<", command=lambda: back(image_number - 1))

    if image_number == 5:
        button_forward = Button(root, text=">>", state=DISABLED)

    result_img.grid(row=1, column=1, ipady=5)
    button_back.grid(row=3, column=0)
    button_forward.grid(row=3, column=2)


def back(image_number):
    global result_img
    global button_forward
    global button_back

    result_img.grid_forget()
    result_img = Label(image=image_list[image_number - 1])
    button_forward = Button(root, text=">>", command=lambda: forward(image_number + 1))
    button_back = Button(root, text="<<", command=lambda: back(image_number - 1))

    if image_number == 1:
        button_back = Button(root, text="<<", state=DISABLED)

    result_img.grid(row=1, column=1, ipady=5)
    button_back.grid(row=3, column=0)
    button_forward.grid(row=3, column=2)


button_back = Button(root, text="<<", command=back, state=DISABLED)
button_exit = Button(root, text="Exit Program", command=root.quit)
button_forward = Button(root, text=">>", command=lambda: forward(2))

button_back.grid(row=3, column=0)
button_exit.grid(row=3, column=1)
button_forward.grid(row=3, column=2)
show_confusion_matrix()

#def show_visualization():

root.mainloop()