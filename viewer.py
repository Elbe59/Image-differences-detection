import os
from tkinter import *
from PIL import ImageTk, Image
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from tkinter import messagebox

root = Tk()
root.title('Results visualization')

image_list = []
result_img = ""
original_img = ""
image_ref = ""
legende = ""
button_forward = ""
button_back = ""
button_exit = ""
box_matrice_legende = Label(root)
box_matrice_legende.grid(row=2, column=3)


def add_original_image(adress):
    global image_ref
    image_ref = ImageTk.PhotoImage(Image.open(adress).resize((750, 500), Image.ANTIALIAS))


def add_results_image(adress: str, image_name, confusion_matrix, data_result):
    global image_list
    my_img = ImageTk.PhotoImage(Image.open(adress).resize((750, 500), Image.ANTIALIAS))
    image_list.append(
        {"adress": adress, "image_name": image_name, "image": my_img, "confusion_matrix": confusion_matrix,
         "accuracy": data_result[0], "recall": data_result[1], "precision": data_result[2], "f1_score": data_result[3]})


def flatten(t):
    return [item for sublist in t for item in sublist]


def show_confusion_matrix(cf_matrix):
    plt.figure(figsize=(4, 4), dpi=70)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    flatten(cf_matrix)]

    group_percentages = ["{0:0.0%}".format(value) for value in
                         flatten(cf_matrix) / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Confusion Matrix of this image')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    if not os.path.exists('./temp/'):
        os.makedirs('./temp/')

    plt.savefig('./temp/confusion_matrix.png')
    image = ImageTk.PhotoImage(Image.open("./temp/confusion_matrix.png"))

    result_img = Label(box_matrice_legende, image=image)
    result_img.image = image  # !!! It is necessary to keep a reference of the image, otherwise the image won't display
    result_img.grid(row=0, column=0, ipady=5)


def navigate_forward_back(image_number):
    global legende

    img = ImageTk.PhotoImage(Image.open("./ressources/GUI/legende.png").resize((150, 100), Image.ANTIALIAS))
    legende = Label(box_matrice_legende, image=img)
    legende.image = img
    legende.grid(row=0, column=1, ipady=5)
    display_result_image(image_number)
    update_navigation_bar(image_number)


def update_navigation_bar(image_number):
    global button_forward
    global button_back
    global button_exit

    button_exit = Button(root, text="Exit Program", command=root.quit, font=('Arial', 10), background="red")

    if image_number == 0:
        button_back = Button(root, text="Previous", state=DISABLED, font=('Arial', 10))
    else:
        button_back = Button(root, text="Previous", command=lambda: navigate_forward_back(image_number - 1),
                             font=('Arial', 10), background="green")

    if image_number == len(image_list) - 1:
        button_forward = Button(root, text="Next", state=DISABLED, font=('Arial', 10))
    else:
        button_forward = Button(root, text="Next", command=lambda: navigate_forward_back(image_number + 1),
                                font=('Arial', 10), background="green")

    button_back.grid(row=4, column=0, pady=(0, 5), padx=(5, 0))
    button_exit.grid(row=4, column=2, pady=(0, 5))
    button_forward.grid(row=4, column=5, pady=(0, 5), padx=(0, 5))


def display_data_results(image_number):
    accuracy = image_list[image_number]["accuracy"]
    recall = image_list[image_number]["recall"]
    precision = image_list[image_number]["precision"]
    f1_score = image_list[image_number]["f1_score"]

    box = Label(root, text="Résultats")
    box.grid(row=2, column=1, ipadx=0)
    titre = Label(box, text="Résultats : \n", font=("Arial", 18))
    accuracy = Label(box, text="Accuracy : " + str(accuracy) + "%", font=("Arial", 15))
    # accuracy, recall, precision, f1_score
    recall = Label(box, text="Recall : " + str(recall) + "%", font=("Arial", 15))
    precision = Label(box, text="Precision : " + str(precision) + "%", font=("Arial", 15))
    f1_score = Label(box, text="f1_score : " + str(f1_score) + "%", font=("Arial", 15))
    titre.grid(row=0, column=0, ipadx=75)
    accuracy.grid(row=1, column=0, ipadx=75)
    recall.grid(row=2, column=0, ipadx=75)
    precision.grid(row=3, column=0, ipadx=75)
    f1_score.grid(row=4, column=0, ipadx=75)


def display_result_image(image_number):
    global result_img

    result_img = Label(image=image_list[image_number]["image"])
    show_confusion_matrix(image_list[image_number]["confusion_matrix"])
    display_data_results(image_number)
    result_img.grid(row=1, column=3, pady=5)
    label_result = Label(root, text="Result Image : " + image_list[image_number]["image_name"], font=("Arial", 10))
    label_result.grid(row=0, column=3, pady=5)
    result_img.grid(row=1, column=3, pady=5)


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.quit()


def show_visualization():
    global original_img

    original_img = Label(image=image_ref)
    original_img.grid(row=1, column=1, pady=5)
    label_original = Label(root, text="Original Image", font=("Arial", 10))
    label_original.grid(row=0, column=1, pady=5)

    navigate_forward_back(0)
    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()
