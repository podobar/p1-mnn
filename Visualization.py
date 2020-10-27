import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

class Visualization:

    @staticmethod
    def draw_2D_result_plot_classification(predictions, test_data, positives):
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f'Classification results: {(positives*100.0)/len(test_data)}%')
        fig.tight_layout(pad=3.0)

        x = [row[0] for row in test_data]
        y = [row[1] for row in test_data]
        axs[0].set_title('Predictions')
        axs[1].set_title('Test set')

        colors_predicted = predictions
        colors_test = [row[2] for row in test_data]

        axs[0].scatter(x, y, c=colors_predicted, cmap='cool', marker=".")
        axs[1].scatter(x, y, c=colors_test, cmap='cool', marker=".")
        fig.show()

    @staticmethod
    def draw_2D_plot(data, plot_title):
        x = [row[0] for row in data]
        y = [row[1] for row in data]
        colors = [row[2] for row in data]
        plt.scatter(x, y, c=colors, cmap='cool', marker=".")
        plt.colorbar(label="classification")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(plot_title)
        plt.show()

    @staticmethod
    def draw_neural_network(network, title, isInitial):
        # _plotCount =
        # fig, axs = plt.subplots(2*len(network.Biases) + 1)
        # fig.suptitle(title)
        # #Input layer
        # axs[0].plot()
        # axs[0].title('Input layer weights')
        # #Hidden layers
        # for i in range(1, 2*len(network.Biases), 2):
        #     axs[i].title(f'Hidden layer ({i}) weights ')
        #     axs[i].title(f'Hidden layer ({i}) biases')
        #
        #
        # #Output layer
        # axs[len(network.Biases)].title('Output layer weights')
        # axs[len(network.Biases)].title('Output layer biases')
        return