from Activation import Activation
from CostFunctions import CostFunctions
from NeuralNetwork import NeuralNetwork
from Visualization import Visualization
import csv
import numpy as np
import logging

modes = {1: "Classification", 2: "Regression"}
log_file_path = "logs\\history.log"


def load_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        _data = list(reader)

    for index in range(1, len(_data)):
        _data[index] = [float(item) for item in _data[index]]

    return _data


def modify_for_multiclass(data, n_class):
    for i in range(len(data)):
        class_vector = np.zeros(n_class)
        class_vector[int(data[i][input_size]) - 1] = 1
        data[i] = data[i][0:input_size] + list(class_vector)
    return data


def modify_data_for_classification(data, n_class, samples_per_class, have_to_mix):
    if not have_to_mix:
        return data
    data_reordered = list()
    for i in range(samples_per_class):
        for j in range(n_class):
            data_reordered.append(data[i + j * samples_per_class])

    return data_reordered


def split_dataset(data, factor):
    train_set = list()
    val_set = list()
    for i in range(1, len(data)):
        if i % int(1 / factor) == 0:
            val_set.append(data[i])
        else:
            train_set.append(data[i])
    return train_set, val_set


def classification_evaluate(network: NeuralNetwork, test_data, n_class):
    positives = 0
    predictions = list()
    for i in range(len(test_data)):
        result = network.forward_propagation(test_data[i][:input_size], activation, out_activation)
        predicted = np.argmax(result) + 1
        actual = int(test_data[i][input_size])
        predictions.append(predicted)
        if predicted == actual:
            positives += 1
    Visualization.draw_2D_result_plot_classification(predictions, test_data, positives)
    return positives


def regression_evaluate(network: NeuralNetwork, test_data):
    actuals = list()
    predictions = list()
    for i in range(len(test_data)):
        result = network.forward_propagation(test_data[i][0:input_size], activation, out_activation)
        actual = float(test_data[i][input_size])
        actuals.append(actual)
        predictions.append(result)

    Visualization.draw_2D_result_plot_regression(predictions, test_data, CostFunctions.MSE(actuals, predictions))
    return CostFunctions.MSE(actuals, predictions)


if __name__ == "__main__":

    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    logging.info('\nProgram started\n')

    # wybieramy rozmiar danych - wybierane są odpowiednie pliki
    # oraz przydaje się do odpowiedniego ustawienia danych przy klasyfikacji do więcej niż 2 klas
    samples_per_class = 500
    train_data_filename = "classification\\data.simple.train."+str(samples_per_class)+".csv"
    test_data_filename = "classification\\data.simple.test."+str(samples_per_class)+".csv"
    have_to_mix = False

    problem = 1  # zgodnie z modes (patrz powyżej)
    n_class = 2  # liczba klas w przypadku problemu klasyfikacji

    iterations = 500  # iteracje uczenia
    learning_factor = 1  # współczynnik uczenia
    input_size = 2
    output_size = n_class if problem == 1 else 1

    multi_class_fl = (problem == 1) & (output_size > 1)

    activation = Activation.sigmoid
    derivative = Activation.sigmoid_derivative
    out_activation = Activation.softmax
    out_derivative = Activation.softmax_derivative
    cost_gradient = CostFunctions.MSE_gradient

    network = NeuralNetwork([input_size, 10, output_size])

    #Visualization.write_out_neural_network_params(network)

    rdata = load_csv(train_data_filename)

    if modes[problem] == "Classification":
        data = modify_data_for_classification(rdata[1:], n_class, samples_per_class, have_to_mix)
        data = modify_for_multiclass(data, n_class)
    #train_set, val_set = split_dataset(data, val_set_factor)

    network.learn(input_size, data, [], activation, out_activation, derivative, out_derivative,
                  cost_gradient, learning_factor, iterations)

    #Visualization.write_out_neural_network_params(network)

    test_data = load_csv(test_data_filename)

    if modes[problem] == "Classification":
        Visualization.draw_2D_plot(rdata[1:], 'Training data [raw]')

        positives = classification_evaluate(network, test_data[1:], n_class)
        total_accuracy = positives/(len(test_data)-1)

        print("Total accuracy:", total_accuracy)

    if modes[problem] == "Regression":
        mean_square_error = regression_evaluate(network, test_data[1:])
        print("Mean square error:", mean_square_error)


