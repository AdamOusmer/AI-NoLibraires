""" Programme qui s'occupe de la récupération de la donnée à analyser et l'affichage du GUI"""

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import prediction

""" 
copyrights:
Adam Ousmer and Sophiane Labrecque

ESP - Collège de Maisonneuve

Present to :
Anik Soulière
"""

""" --------- GUI --------- """


def graphique(image):
    plt.imshow(image, cmap="Greys_r", interpolation="None", vmax=255, vmin=0)  # Création du graphique

    """ Enlever les axes et les graduations """
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.savefig(f"Matrice.png", transparent=True, dpi=200)  # Enregistrement des images
    plt.cla()


""" Création des constante et de la fenêtre """
background_color = "#212121"
foreground_color = "#dedede"
list_selector = np.linspace(0, 255, 256, dtype=int)
picture = np.ones((4, 4)) * 255
root = tk.Tk()
root.title("Ouali")
root.geometry("1500x2560")
root.config()

# Label de titre

label_title = tk.Label(root, text="IA - Reconnaissance de symboles", font=("Lucida Grande", 20))
label_title.pack()

# Frame pour séparation en 2
frame_main = tk.Frame(root)

# Frame pour la sélection de l'image
frame_selection = tk.Frame(frame_main)
frame_visualisation = tk.Frame(frame_main)

label_subtitle = tk.Label(frame_selection, text="Veuillez entrer les couleurs de chaque case de la matrice à analyser.",
                          font=("Lucida Grande", 15))
label_subtitle.grid(row=0, columnspan=4)

label_answer = tk.Label(frame_selection, text="", font=("Lucida Grande", 15))
label_answer.grid(row=6, columnspan=4)

# Canva pour l'image
width = 740
height = 740
picture_visualisation = tk.PhotoImage(file="MatriceAffichageInitiale.png")
canvas = tk.Canvas(frame_visualisation, width=width, height=height)
picture_container = canvas.create_image(width / 2, height / 2, image=picture_visualisation)
canvas.pack(expand=True)


def display():
    graphique(picture)
    picture_visualisation.config(file="Matrice.png")
    canvas.update()


def btn_analyse_action():
    picture_vector = picture.flatten()  # Traitement de l'image en vecteur

    result = prediction.prediction(picture_vector)  # Passage du vecteur à l'IA

    # Traitement du résultat pour l'affichage
    letter = result[0]
    certainty = result[1] * 100

    label_answer.config(text=f"La matrice entrée est de classe {letter}.\n {certainty}% de certitude.")  # Affichage


# Création des sélecteurs

""" Action des sélecteurs """


def value0_0(a):
    picture[0, 0] = var0_0.get()
    display()
    print(picture)


def value0_1(a):
    picture[0, 1] = var0_1.get()
    display()
    print(picture)


def value0_2(a):
    picture[0, 2] = var0_2.get()
    display()
    print(picture)


def value0_3(a):
    picture[0, 3] = var0_3.get()
    display()
    print(picture)


def value1_0(a):
    picture[1, 0] = var1_0.get()
    display()
    print(picture)


def value1_1(a):
    picture[1, 1] = var1_1.get()
    display()
    print(picture)


def value1_2(a):
    picture[1, 2] = var1_2.get()
    display()
    print(picture)


def value1_3(a):
    picture[1, 3] = var1_3.get()
    display()
    print(picture)


def value2_0(a):
    picture[2, 0] = var2_0.get()
    display()
    print(picture)


def value2_1(a):
    picture[2, 1] = var2_1.get()
    display()
    print(picture)


def value2_2(a):
    picture[2, 2] = var2_2.get()
    display()
    print(picture)


def value2_3(a):
    picture[2, 3] = var2_3.get()
    display()
    print(picture)


def value3_0(a):
    picture[3, 0] = var3_0.get()
    display()
    print(picture)


def value3_1(a):
    picture[3, 1] = var3_1.get()
    display()
    print(picture)


def value3_2(a):
    picture[3, 2] = var3_2.get()
    display()
    print(picture)


def value3_3(a):
    picture[3, 3] = var3_3.get()
    display()
    print(picture)


var0_0 = tk.StringVar()
var0_0.set(list_selector[255])
selector0_0 = tk.OptionMenu(frame_selection, var0_0, *list_selector, command=value0_0)
selector0_0.grid(row=1, column=0, padx=10, pady=10)

var0_1 = tk.StringVar()
var0_1.set(list_selector[255])
selector0_1 = tk.OptionMenu(frame_selection, var0_1, *list_selector, command=value0_1)
selector0_1.grid(row=1, column=1, padx=10, pady=10)

var0_2 = tk.StringVar()
var0_2.set(list_selector[255])
selector0_2 = tk.OptionMenu(frame_selection, var0_2, *list_selector, command=value0_2)
selector0_2.grid(row=1, column=2, padx=10, pady=10)

var0_3 = tk.StringVar()
var0_3.set(list_selector[255])
selector0_3 = tk.OptionMenu(frame_selection, var0_3, *list_selector, command=value0_3)
selector0_3.grid(row=1, column=3, padx=10, pady=10)

var1_0 = tk.StringVar()
var1_0.set(list_selector[255])
selector1_0 = tk.OptionMenu(frame_selection, var1_0, *list_selector, command=value1_0)
selector1_0.grid(row=2, column=0, padx=10, pady=10)

var1_1 = tk.StringVar()
var1_1.set(list_selector[255])
selector1_1 = tk.OptionMenu(frame_selection, var1_1, *list_selector, command=value1_1)
selector1_1.grid(row=2, column=1, padx=10, pady=10)

var1_2 = tk.StringVar()
var1_2.set(list_selector[255])
selector1_2 = tk.OptionMenu(frame_selection, var1_2, *list_selector, command=value1_2)
selector1_2.grid(row=2, column=2, padx=10, pady=10)

var1_3 = tk.StringVar()
var1_3.set(list_selector[255])
selector1_3 = tk.OptionMenu(frame_selection, var1_3, *list_selector, command=value1_3)
selector1_3.grid(row=2, column=3, padx=10, pady=10)

var2_0 = tk.StringVar()
var2_0.set(list_selector[255])
selector2_0 = tk.OptionMenu(frame_selection, var2_0, *list_selector, command=value2_0)
selector2_0.grid(row=3, column=0, padx=10, pady=10)

var2_1 = tk.StringVar()
var2_1.set(list_selector[255])
selector2_1 = tk.OptionMenu(frame_selection, var2_1, *list_selector, command=value2_1)
selector2_1.grid(row=3, column=1, padx=10, pady=10)

var2_2 = tk.StringVar()
var2_2.set(list_selector[255])
selector2_2 = tk.OptionMenu(frame_selection, var2_2, *list_selector, command=value2_2)
selector2_2.grid(row=3, column=2, padx=10, pady=10)

var2_3 = tk.StringVar()
var2_3.set(list_selector[255])
selector2_3 = tk.OptionMenu(frame_selection, var2_3, *list_selector, command=value2_3)
selector2_3.grid(row=3, column=3, padx=10, pady=10)

var3_0 = tk.StringVar()
var3_0.set(list_selector[255])
selector3_0 = tk.OptionMenu(frame_selection, var3_0, *list_selector, command=value3_0)
selector3_0.grid(row=4, column=0, padx=10, pady=10)

var3_1 = tk.StringVar()
var3_1.set(list_selector[255])
selector3_1 = tk.OptionMenu(frame_selection, var3_1, *list_selector, command=value3_1)
selector3_1.grid(row=4, column=1, padx=10, pady=10)

var3_2 = tk.StringVar()
var3_2.set(list_selector[255])
selector3_2 = tk.OptionMenu(frame_selection, var3_2, *list_selector, command=value3_2)
selector3_2.grid(row=4, column=2, padx=10, pady=10)

var3_3 = tk.StringVar()
var3_3.set(list_selector[255])
selector3_3 = tk.OptionMenu(frame_selection, var3_3, *list_selector, command=value3_3)
selector3_3.grid(row=4, column=3, padx=10, pady=10)

# Fin création des sélecteurs

# Btn
btn_analyse = tk.Button(frame_selection, text="Analyser", font=("Lucida Grande", 15), fg=background_color,
                        command=btn_analyse_action)
btn_analyse.grid(row=5, columnspan=4, pady=10)

# Affichage du root
frame_selection.grid(row=0, column=0, padx=50, pady=50)
frame_visualisation.grid(row=0, column=2, padx=10, pady=50)
frame_main.pack(expand=True)
root.mainloop()
