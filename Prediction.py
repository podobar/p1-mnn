from Activation import Activation
from CostFunctions import CostFunctions
from NeuralNetwork import NeuralNetwork
from Visualization import Visualization
from Scaler import Scaler
import csv
import numpy as np
import logging

modes = {1: "Classification", 2: "Regression"}
scale_mode = {1: "std", 2: "norm", 3: "none"}

log_file_path = "logs\\history.log"


def load_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        _data = list(reader)

    for index in range(1, len(_data)):
        _data[index] = [float(item) for item in _data[index]]

    return _data


def scale(data, scaler: Scaler, mode, rescaling):
    if scale_mode[mode] == "std":
        fun = scaler.standardize if not rescaling else scaler.unstandardize
    else:
        if scale_mode[mode] == "norm":
            fun = scaler.normalize if not rescaling else scaler.unnormalize
        else:
            fun = Scaler.identity

    return fun(data)


def mix_data_for_classification(data, n_class, samples_per_class):
    data_reordered = list()
    for i in range(samples_per_class):
        for j in range(n_class):
            data_reordered.append(data[i + j * samples_per_class])

    return data_reordered


def modify_for_multiclass(data, n_class):
    data_modified = list()
    for i in range(len(data)):
        class_vector = np.zeros(n_class)
        class_vector[int(data[i][input_size]) - 1] = 1
        data_modified.append(data[i][0:input_size] + list(class_vector))
    return data_modified


def modify_data_for_regression(data, in_scaler, out_scaler):
    inputs = [d[0] for d in data]
    outputs = [d[1] for d in data]

    in_scaler.fit(inputs)
    out_scaler.fit(outputs)

    data = [[scale(d[0], in_scaler, in_scaling, False), scale(d[1], out_scaler, out_scaling, False)]
            for d in data]
    return data, in_scaler, out_scaler

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
        input_data = scale(test_data[i][0], in_scaler, in_scaling, False)
        result = network.forward_propagation([input_data], activation, out_activation)
        actual = float(test_data[i][1])
        actuals.append(actual)
        predictions.append(scale(result, out_scaler, out_scaling, True))

    Visualization.draw_2D_result_plot_regression(predictions, test_data, CostFunctions.MSE(actuals, predictions))
    return CostFunctions.MSE(actuals, predictions)


if __name__ == "__main__":

    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    logging.info('\nProgram started\n')

    # wybieramy rozmiar danych - wybierane są odpowiednie pliki
    # oraz przydaje się do odpowiedniego ustawienia danych przy klasyfikacji do więcej niż 2 klas
    samples_per_class = 1000
    train_data_filename = "regression\\data.activation.train."+str(samples_per_class)+".csv"
    test_data_filename = "regression\\data.activation.test."+str(samples_per_class)+".csv"
    have_to_mix = False

    problem = 2  # zgodnie z modes (patrz powyżej)
    n_class = 3  # liczba klas w przypadku problemu klasyfikacji

    iterations = 2000  # iteracje uczenia
    learning_factor = 0.000001  # współczynnik uczenia
    input_size = 1
    output_size = n_class if problem == 1 else 1

    multi_class_fl = (problem == 1) & (output_size > 1)

    network = NeuralNetwork([input_size, 10, output_size])
    in_scaler = Scaler()
    out_scaler = Scaler()

    activation = Activation.tanH
    derivative = Activation.tanH_derivative
    out_activation = Activation.linear
    out_derivative = Activation.linear_derivative
    cost = CostFunctions.MSE
    cost_gradient = CostFunctions.kullback_gradient
    in_scaling = 3
    out_scaling = 1

    #Visualization.write_out_neural_network_params(network)

    rdata = load_csv(train_data_filename)[1:]

    if modes[problem] == "Classification":
        if have_to_mix:
            rdata = mix_data_for_classification(rdata, n_class, samples_per_class)
        data = modify_for_multiclass(rdata, n_class)

    if modes[problem] == "Regression":
        data, in_scaler, out_scaler = modify_data_for_regression(rdata[1:], in_scaler, out_scaler)

    network.learn(input_size, data, activation, out_activation, derivative, out_derivative,
                  cost_gradient, cost, learning_factor, iterations)

    #Visualization.write_out_neural_network_params(network)

    test_data = load_csv(test_data_filename)

    if modes[problem] == "Classification":
        Visualization.draw_2D_plot(rdata, 'Training data [raw]')

        positives = classification_evaluate(network, test_data[1:], n_class)
        total_accuracy = positives/(len(test_data)-1)

        print("Total accuracy:", total_accuracy)

    if modes[problem] == "Regression":
        mean_square_error = regression_evaluate(network, test_data[1:])
        print("Mean square error:", mean_square_error)


