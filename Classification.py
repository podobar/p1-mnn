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


def binary_classification_prediction(network: NeuralNetwork, test_data):
    confusion_matrix = np.zeros([2, 2])

    for i in range(len(test_data)):
        result = network.forward_propagation(test_data[i][0:input_size], activation, out_activation)
        predicted = 2 if result >= 1.5 else 1
        actual = int(test_data[i][input_size])
        confusion_matrix[predicted - 1][actual - 1] += 1

    return confusion_matrix


if __name__ == "__main__":

    train_data_filename = "classification\\data.simple.train.100.csv"
    test_data_filename = "classification\\data.simple.test.100.csv"

    problem = 1
    learning_factor = 1
    input_size = 2
    output_size = 1

    activation = Activation.relu
    out_activation = Activation.sigmoid
    derivative = Activation.relu_derivative
    out_derivative = Activation.sigmoid_derivative

    network = NeuralNetwork(3, [input_size, 2, 2, output_size])

    data = load_csv(train_data_filename)

    for ind in range(1, len(data)):
        network.forward_propagation(data[ind][0:input_size], activation, out_activation)
        network.back_propagation_error(data[ind][input_size:len(data)], derivative,
                                       out_derivative, CostFunctions.MSE_gradient)
        network.update_weights(learning_factor)

    test_data = load_csv(test_data_filename)

    if modes[problem] == "Classification":
        #multiple_class_classification_prediction()
        confusion_matrix = np.array(binary_classification_prediction(network, test_data[1:]))

        n = len(confusion_matrix)
        positives = np.array([confusion_matrix[i][i] for i in range(n)])

        total_accuracy = positives.sum()/confusion_matrix.sum()
        precisions = [positives[i]/confusion_matrix[i].sum() for i in range(n)]
        recalls = [positives[i]/confusion_matrix[i].transpose().sum() for i in range(n)]

        print("Total accuracy:", total_accuracy)
        for i in range(n):
            print("Precision_"+str(i+1), precisions[i])
            print("Recall_" + str(i+1), recalls[i])




