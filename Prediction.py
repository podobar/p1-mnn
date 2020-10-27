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


def classification_evaluate(network: NeuralNetwork, test_data, binary):
    positives = 0
    predictions = list()
    for i in range(len(test_data)):
        result = network.forward_propagation(test_data[i][0:input_size], activation, out_activation)
        predicted = (2 if result[0] >= 1.5 else 1) if binary else np.argmax(result) + 1
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
    train_data_filename = "classification\\data.three_gauss.train.500.csv"
    test_data_filename = "classification\\data.three_gauss.test.500.csv"

    train_data_filename = "regression\\data.activation.train.500.csv"
    test_data_filename = "regression\\data.activation.test.500.csv"
    problem = 1
    problem = 2
    val_set_factor = 0.2

    learning_factor = 0.1
    input_size = 2
    input_size = 1
    output_size = 3
    output_size = 1

    multi_class_fl = (problem == 1) & (output_size > 1)

    activation = Activation.relu
    derivative = Activation.relu_derivative
    out_activation = Activation.softmax
    out_derivative = Activation.softmax_derivative
    cost_gradient = CostFunctions.MSE_gradient
    loss_function = CostFunctions.MSE

    network = NeuralNetwork(3, [input_size, 1, output_size])
    Visualization.write_out_neural_network_params(network, isInitial=True)
    data = load_csv(train_data_filename)
    if multi_class_fl:
        for i in range(1, len(data)):
            class_vector = np.zeros(output_size)
            class_vector[int(data[i][input_size])-1] = 1
            data[i] = data[i][0:input_size] + list(class_vector)

    train_set = list()
    val_set = list()
    for i in range(1, len(data)):
        if i % int(1 / val_set_factor) == 0:
            val_set.append(data[i])
        else:
            train_set.append(data[i])

    network.learn(input_size, train_set, val_set, activation, out_activation, derivative, out_derivative,
                  cost_gradient, loss_function, learning_factor)
    Visualization.write_out_neural_network_params(network)
    logging.info(f"\n=========================\nNetwork trained\n")
    test_data = load_csv(test_data_filename)

    if modes[problem] == "Classification":
        Visualization.draw_2D_plot(data[1:], 'Training data [raw]')
        positives = classification_evaluate(network, test_data[1:], output_size == 1)

        total_accuracy = positives/(len(test_data)-1)
        # precisions = [positives[i]/confusion_matrix[i].sum() for i in range(n)]
        # recalls = [positives[i]/confusion_matrix[i].transpose().sum() for i in range(n)]

        print("Total accuracy:", total_accuracy)
        # for i in range(n):
        #     print("Precision_"+str(i+1), precisions[i])
        #     print("Recall_" + str(i+1), recalls[i])

    if modes[problem] == "Regression":
        mean_square_error = regression_evaluate(network, test_data[1:])

        print("Mean square error:", mean_square_error)


