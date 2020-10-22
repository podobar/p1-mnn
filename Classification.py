from Activation import Activation
from CostFunctions import CostFunctions
from NeuralNetwork import NeuralNetwork
import csv
import numpy as np

modes = {1: "Classification", 2: "Regression"}


def load_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        _data = list(reader)

    for index in range(1, len(_data)):
        _data[index] = [float(item) for item in _data[index]]

    return _data


def classification_evaluate(network: NeuralNetwork, test_data, binary):
    positives = 0
    for i in range(len(test_data)):
        result = network.forward_propagation(test_data[i][0:input_size], activation, out_activation)
        predicted = (2 if result[0] >= 1.5 else 1) if binary else np.argmax(result) + 1
        actual = int(test_data[i][input_size])
        if predicted == actual:
            positives += 1
    return positives


if __name__ == "__main__":

    train_data_filename = "classification\\data.three_gauss.train.100.csv"
    test_data_filename = "classification\\data.three_gauss.test.100.csv"

    problem = 1

    learning_factor = 0.15
    input_size = 2
    output_size = 3
    multi_class_fl = (problem == 1) & (output_size > 1)

    activation = Activation.relu
    derivative = Activation.relu_derivative
    out_activation = Activation.softmax
    out_derivative = Activation.softmax_derivative
    cost_gradient = CostFunctions.MSE_gradient
    loss_function = CostFunctions.MSE

    network = NeuralNetwork(1, [input_size, 7, output_size])

    data = load_csv(train_data_filename)

    if multi_class_fl:
        for i in range(1, len(data)):
            class_vector = np.zeros(output_size)
            class_vector[int(data[i][input_size])-1] = 1
            data[i] = data[i][0:input_size] + list(class_vector)

    train_set = list()
    val_set = list()
    for i in range(1, len(data), 2):
        train_set.append(data[i])
        val_set.append((data[i+1]))

    network.learn(input_size, train_set, val_set, activation, out_activation, derivative, out_derivative,
                  cost_gradient, loss_function, learning_factor)

    test_data = load_csv(test_data_filename)

    if modes[problem] == "Classification":

        positives = classification_evaluate(network, test_data[1:], output_size == 1)

        total_accuracy = positives/(len(test_data)-1)
        # precisions = [positives[i]/confusion_matrix[i].sum() for i in range(n)]
        # recalls = [positives[i]/confusion_matrix[i].transpose().sum() for i in range(n)]

        print("Total accuracy:", total_accuracy)
        # for i in range(n):
        #     print("Precision_"+str(i+1), precisions[i])
        #     print("Recall_" + str(i+1), recalls[i])




