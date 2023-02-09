""" Programme de deep learning avec des tailles de couches spécifiques """

import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split as sk_split
from tqdm import tqdm

""" 
copyrights:
Adam Ousmer and Sophiane Labrecque

ESP - Collège de Maisonneuve

Present to :
Anik Soulière
"""

""" ---------  Fonctions  -------- """


def writer_csv(x_csv, y_csv, file_name_csv):
    """ Fonction pour écrire les données et les étiquettes dans un fichier csv """

    csvFile = open(f"{file_name_csv}.csv", "w", newline="")  # Ouverture du fichier

    # Paramétrage de l'écriture
    fieldnames = ["Data", "X", "Y"]
    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(len(x_csv)):  # Boucle d'écriture
        writer.writerow({"Data": " ", "X": " ", "Y": " "})
        writer.writerow({"Data": i, "X": x_csv[i], "Y": y_csv[i]})

    csvFile.close()  # Fermeture du fichier


def writer_txt(x_txt, subtitle, file_name_txt, open_type, x_txt_type=" "):
    """ Fonction pour écrire une donnée dans un fichier txt
        Utilisation pour les poids et les biais """

    txt_file = open(f"{file_name_txt}.txt", f"{open_type}")  # Ouverture du fichier

    txt_file.write(f"{subtitle}\n")

    if x_txt_type == "matrix":
        txt_file.write("[")
        for i_lines in range(x_txt.shape[0]):  # Écriture
            txt_file.write("\n")
            txt_file.write("],\n[ ")
            for i_columns in range(x_txt.shape[1]):
                txt_file.write(f"{x_txt[i_lines, i_columns]},")
        txt_file.write("]]\n")

    elif x_txt_type == "vector":
        txt_file.write("[")
        for i_vector in range(len(x_txt)):
            txt_file.write(f"{x_txt[i_vector]},")
        txt_file.write("]\n")

    else:
        txt_file.write(f"{x_txt}\n")

    txt_file.close()


def graphic_plot(axis, color="blue", save_plot=True, title_plot="Title"):
    """Fonction qui crée les graphiques """

    plt.plot(axis, color=color)

    if save_plot:
        plt.savefig(f"{title_plot}.png", dpi=300)
        plt.cla()


def graphic_weight(weight_graphic, color="inferno", save_weight=True, title_weight="Title"):
    """ Fonction qui crée les images pour visualiser les poids """

    plt.imshow(weight_graphic, cmap=color, interpolation="None")

    if save_weight:
        plt.savefig(f"{title_weight}.png", dpi=300, transparent=True)
        plt.cla()


def random_number(lower_bound, upper_bound):
    """ Fonction qui renvoie un nombre situé entre 2 bornes tel que [min, max[ """

    return np.random.randint(lower_bound, upper_bound)


def dataset_generator(size=120000):
    """ Fonction qui génère les images sous forme de vecteurs """

    dataset_generate = []
    dataset_labels = []

    for i_generator in tqdm(range(size)):
        # Matrice A
        vector = [random_number(50, 256) / 255, random_number(0, 36) / 255, random_number(50, 256) / 255,
                  random_number(50, 256) / 255,
                  random_number(0, 36) / 255, random_number(50, 256) / 255, random_number(0, 36) / 255,
                  random_number(50, 256) / 255,
                  random_number(50, 256) / 255, random_number(0, 36) / 255, random_number(50, 256) / 255,
                  random_number(0, 36) / 255,
                  random_number(50, 256) / 255, random_number(50, 256) / 255, random_number(0, 36) / 255,
                  random_number(50, 256) / 255]

        dataset_generate.append(vector)
        dataset_labels.append([1, 0, 0, 0, 0])

        # Matrice B
        vector = [random_number(0, 36) / 255, random_number(50, 256) / 255, random_number(50, 256) / 255,
                  random_number(50, 256) / 255,
                  random_number(50, 256) / 255, random_number(0, 36) / 255, random_number(50, 256) / 255,
                  random_number(50, 256) / 255,
                  random_number(50, 256) / 255, random_number(50, 256) / 255, random_number(0, 36) / 255,
                  random_number(50, 256) / 255,
                  random_number(50, 256) / 255, random_number(50, 256) / 255, random_number(50, 256) / 255,
                  random_number(0, 36) / 255]

        dataset_generate.append(vector)
        dataset_labels.append([0, 1, 0, 0, 0])

        # Matrice C
        vector = [random_number(50, 256) / 255, random_number(50, 256) / 255, random_number(50, 256) / 255,
                  random_number(0, 36) / 255,
                  random_number(50, 256) / 255, random_number(50, 256) / 255, random_number(0, 36) / 255,
                  random_number(50, 256) / 255,
                  random_number(50, 256) / 255, random_number(0, 36) / 255, random_number(50, 256) / 255,
                  random_number(50, 256) / 255,
                  random_number(0, 36) / 255, random_number(50, 256) / 255, random_number(50, 256) / 255,
                  random_number(50, 256) / 255]

        dataset_generate.append(vector)
        dataset_labels.append([0, 0, 1, 0, 0])

        # Matrice D
        vector = [random_number(50, 256) / 255, random_number(50, 256) / 255, random_number(50, 256) / 255,
                  random_number(50, 256) / 255,
                  random_number(0, 36) / 255, random_number(50, 256) / 255, random_number(50, 256) / 255,
                  random_number(0, 36) / 255,
                  random_number(50, 256) / 255, random_number(0, 36) / 255, random_number(0, 36) / 255,
                  random_number(50, 256) / 255,
                  random_number(50, 256) / 255, random_number(50, 256) / 255, random_number(50, 256) / 255,
                  random_number(50, 256) / 255]

        dataset_generate.append(vector)
        dataset_labels.append([0, 0, 0, 1, 0])

        # Matrice E
        vector = [random_number(50, 256) / 255, random_number(0, 36) / 255, random_number(0, 36) / 255,
                  random_number(50, 256) / 255,
                  random_number(0, 36) / 255, random_number(50, 256) / 255, random_number(50, 256) / 255,
                  random_number(0, 36) / 255,
                  random_number(0, 36) / 255, random_number(50, 256) / 255, random_number(50, 256) / 255,
                  random_number(0, 36) / 255,
                  random_number(50, 256) / 255, random_number(0, 36) / 255, random_number(0, 36) / 255,
                  random_number(50, 256) / 255]

        dataset_generate.append(vector)
        dataset_labels.append([0, 0, 0, 0, 1])

    return dataset_generate, dataset_labels


def initialize_weights_and_biais(x_lines, layer_size=8, output_size=5):
    """ Fonction qui initialise aléatoirement les poids et les biais """

    W1 = np.random.rand(len(x_lines), layer_size)
    b1 = np.random.rand(layer_size)

    W_output = np.random.rand(W1.shape[1], output_size)
    b_output = np.random.rand(output_size)

    weight_initialize = [W1, b1, W_output, b_output]

    return weight_initialize


def sigmoid(z_sigmoid):
    """ Fonction qui calcule la sigmoid"""
    z = 1 / (1 + np.e ** -z_sigmoid)
    return z


def sigmoid_prime(z_sigmoid_prime):
    """ Fonction qui calcule la dérivée de la sigmoid """

    return np.e ** -z_sigmoid_prime / (1 + np.e ** -z_sigmoid_prime) ** 2


def cost(activations_cost, y_cost):
    """ Fonction qui calcule le coût"""

    summation = (np.power((activations_cost - y_cost), 2))

    return np.sum(summation)


def accuracy(activation_accuracy, y_activation):
    """ Fonction qui calcul détermine si la prédiction est correcte """

    index = max(y_activation)

    if max(activation_accuracy) == index:
        return 1
    else:
        return 0


def gradient(old_weights, y_gradient, activation_gradient, x):
    """Fonction qui effectue le calcul du gradient """

    # Récupération des poids et biais
    W1 = old_weights[0]
    b1 = old_weights[1]

    W_output = old_weights[2]
    b_output = old_weights[3]

    z1 = activation_gradient.get("z1")
    a1 = activation_gradient.get("a1")

    z_output = activation_gradient.get("z_output")
    a_output = activation_gradient.get("a_output")

    # Création de matrice de même format que les poids pour enregistrer les gradients
    W1_gradient_value = np.zeros(W1.shape)
    b1_gradient_value = np.zeros(len(b1))

    W_output_gradient_value = np.zeros(W_output.shape)
    b_output_gradient_value = np.zeros(len(b_output))

    for n in range(W_output.shape[0]):
        for m in range(W_output.shape[1]):
            W_output_gradient_value[n, m] = 2 * (a_output[m] - y_gradient[m]) * sigmoid_prime(z_output[m]) * a1[n]

    for m in range(len(b_output)):
        b_output_gradient_value[m] = 2 * (a_output[m] - y_gradient[m]) * sigmoid_prime(z_output[m])

    for i_grad in range(W1.shape[0]):
        for n in range(W1.shape[1]):
            W1_gradient_value[i_grad, n] = np.sum(
                2 * (a_output - y_gradient) * sigmoid_prime(z_output) * W_output[n]) * sigmoid_prime(z1[n]) * x[i_grad]

    for n in range(len(b1)):
        b1_gradient_value[n] = np.sum(
            2 * (a_output - y_gradient) * sigmoid_prime(z_output) * W_output[n]) * sigmoid_prime(z1[n])

    gradient_values = {
        "W1": W1_gradient_value,
        "b1": b1_gradient_value,
        "W_output": W_output_gradient_value,
        "b_output": b_output_gradient_value
    }

    return gradient_values


def correction(weights_to_correct, c_gradients, t_step):
    """ Fonction qui effectue la correction des poids et des biais selon le gradient"""

    W1_corrected = weights_to_correct[0] - (t_step * c_gradients[0])
    b1_corrected = weights_to_correct[1] - (t_step * c_gradients[1])

    W_output_corrected = weights_to_correct[2] - (t_step * c_gradients[2])
    b_output_corrected = weights_to_correct[3] - (t_step * c_gradients[3])

    weights_corrected = [W1_corrected, b1_corrected, W_output_corrected, b_output_corrected]

    return weights_corrected


def normalize(activation):

    normalizing = (np.sum(activation ** 2)) ** (1/2)

    return activation / normalizing # return [a1/normalizing , a2/normalizing, a3/normalizing]


def neural_network_train(x_train, y_train, x_test, y_test, learning_step=0.01, iteration=20000):
    """ Fonction qui contient toutes les étapes de l'entrainement """

    weights = initialize_weights_and_biais(x_train[0])

    cost_train = []
    cost_graphic = []
    cost_test_graphic = []

    accuracy_train = []
    accuracy_test_graphic = []
    accuracy_graphic = []

    gradient_calculated = []

    for i in tqdm(range(iteration)):
        for i_train in range(len(x_train)):
            x = x_train[i_train]
            y = y_train[i_train]

            z1 = np.matmul(x, weights[0]) + weights[1]
            a1 = sigmoid(z1)

            z_output = np.matmul(a1, weights[2]) + weights[3]
            a_output = sigmoid(z_output)

            a_output = normalize(a_output) # [0 0 0 0 0]

            activation = {
                "z1": z1,
                "z_output": z_output,
                "a1": a1,
                "a_output": a_output
            }

            cost_train_specific_data = cost(a_output, y)

            cost_train.append(cost_train_specific_data)

            accuracy_train_specific_data = accuracy(a_output, y)
            accuracy_train.append(accuracy_train_specific_data)

            gradient_calculated.append(gradient(weights, y, activation, x))

            if i_train % 1000 == 0 and i_train != 0:

                graphic_weight(weights[0], title_weight=f"W1_{i_train}_{i}")
                graphic_weight(weights[2], title_weight=f"W_output_{i_train}_{i}")

                gradient_W1 = np.zeros(gradient_calculated[0].get("W1").shape)
                gradient_b1 = np.zeros(len(gradient_calculated[0].get("b1")))
                gradient_W_output = np.zeros(gradient_calculated[0].get("W_output").shape)
                gradient_b_output = np.zeros(len(gradient_calculated[0].get("b_output")))

                for i_gradient in gradient_calculated:
                    gradient_W1 += i_gradient.get("W1")
                    gradient_b1 += i_gradient.get("b1")
                    gradient_W_output += i_gradient.get("W_output")
                    gradient_b_output += i_gradient.get("b_output")

                gradient_average = [gradient_W1 / len(gradient_calculated),
                                    gradient_b1 / len(gradient_calculated),
                                    gradient_W_output / len(gradient_calculated),
                                    gradient_b_output / len(gradient_calculated)]

                weights = correction(weights, gradient_average, learning_step)

                gradient_calculated = []

        if i % 10 == 0:  # Création de graphiques

            sum_accuracy = 0
            for average_accuracy_counter in accuracy_train:
                sum_accuracy += average_accuracy_counter
            average_accuracy = sum_accuracy / len(accuracy_train)

            accuracy_graphic.append(average_accuracy)


            sum_cost = 0
            for average_cost_counter in cost_train:
                sum_cost += average_cost_counter
            average_cost = sum_cost / len(cost_train)

            cost_graphic.append(average_cost)


            cost_train = []
            accuracy_train = []

    # Création et enregistrement des graphiques

    graphic_plot(cost_graphic, color="darkblue", save_plot=False, title_plot="Cost Fonction")
    graphic_plot(cost_test_graphic, color="red", save_plot=True, title_plot="Cost Fonction")

    graphic_plot(accuracy_graphic, color="darkblue", save_plot=False, title_plot="Accuracy Score")
    graphic_plot(accuracy_test_graphic, color="red", save_plot=True, title_plot="Accuracy Score")

    return weights


""" ---------  MAIN  -------- """

print("\n \n")  # Mise en page de la console
print("Generating the dataset. \n")

x_raw, y_raw = dataset_generator(size=20000)

print("\nDataset have been generated. \n")

# Séparation du dataset en train set et test set
x_train_split, x_test_split, y_train_split, y_test_split = sk_split(x_raw, y_raw)

print("Writing the data... \n")

writer_csv(x_raw, y_raw, "Dataset - Raw Data")
writer_csv(x_train_split, y_train_split, "Dataset - Train set")
writer_csv(x_test_split, y_test_split, "Dataset - Test set")

print("Data have been written. \n")
print("Training...\n")

weight_trained = neural_network_train(x_train_split, y_train_split, x_test_split, y_test_split, iteration=100)

print("Training is finish.\n")
print("Writing the trained weights...\n")

writer_txt(weight_trained[0], "W First Layer", "Weight", open_type="w", x_txt_type="matrix")
writer_txt(weight_trained[2], "W Output", "Weight", open_type="a", x_txt_type="matrix")
writer_txt(weight_trained[1], "b First Layer", "Weight", open_type="a", x_txt_type="vector")
writer_txt(weight_trained[3], "b Output", "Weight", open_type="a", x_txt_type="vector")

print("Trained weights have been saved.\n")
print("\n")  # Mise en page de la console

#%%
