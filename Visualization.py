import matplotlib.pyplot as plt
import logging


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
        plt.show()

    @staticmethod
    def draw_2D_result_plot_regression(predictions, test_data, diff):
        x = [row[0] for row in test_data]
        y = [row[1] for row in test_data]
        plt.scatter(x, y, c='black', marker=".", s=1)
        plt.scatter(x, predictions, c='red', marker=".", s=1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Regression results, MSE:{diff}')
        plt.show()

    @staticmethod
    def draw_2D_loss_per_iteration(losses):
        x = [row[0] for row in losses]
        y = [row[1] for row in losses]
        plt.plot(x,y)
        plt.xlabel('Iterations')
        plt.ylabel('Loss on val set')
        plt.title(f'Loss:')
        plt.show()

    @staticmethod
    def draw_2D_plot(data, plot_title):
        x = [row[0] for row in data]
        y = [row[1] for row in data]
        colors = [row[2] for row in data]
        plt.scatter(x, y, c=colors, cmap='cool', marker=".")
        plt.colorbar(label="cls")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(plot_title)
        plt.show()

    @staticmethod
    def write_out_neural_network_params(network):
        # Input layer
        logging.info('Input layer weights:')
        logging.info(network.Weights[0])
        # Hidden layers
        for i in range(1, len(network.Biases)):
            logging.info(f'Hidden layer ({i}) weights: ')
            logging.info(network.Weights[i])
            logging.info(f'Hidden layer ({i}) biases: ')
            logging.info(network.Biases[i-1])
        # Output layer
        logging.info('Output layer biases')
        logging.info(network.Biases[len(network.Biases) - 1])

        #Input layer
        print('Input layer weights:')
        print(network.Weights[0])
        #Hidden layers
        for i in range(1, len(network.Biases)):
            print(f'Hidden layer ({i}) weights: ')
            print(network.Weights[i])
            print(f'Hidden layer ({i}) biases: ')
            print(network.Biases[i-1])
        #Output layer
        print('Output layer biases')
        print(network.Biases[len(network.Biases)-1])
        print("\n\n")
        return

    @staticmethod
    def write_out_neural_network_weights(network):

        # Input layer
        logging.info('Input layer weights:')
        logging.info(network.Weights[0])
        # Hidden layers
        for i in range(1, len(network.Weights)):
            logging.info(f'Hidden layer ({i}) weights: ')
            logging.info(network.Weights[i])

        # Input layer
        print('Input layer weights:')
        print(network.Weights[0])
        # Hidden layers
        for i in range(1, len(network.Weights)):
            print(f'Hidden layer ({i}) weights: ')
            print(network.Weights[i])
        print("\n\n")

    @staticmethod
    def write_out_neuron_errors(network):
        # Input layer
        logging.info('Input layer errors:')
        logging.info(network.Errors[0])
        # Hidden layers
        for i in range(1, len(network.Errors)):
            logging.info(f'Hidden layer ({i}) errors: ')
            logging.info(network.Errors[i])
        # Input layer
        print('Input layer errors:')
        print(network.Errors[0])
        # Hidden layers
        for i in range(1, len(network.Errors)):
            print(f'Hidden layer ({i}) errors: ')
            print(network.Errors[i])
        print("\n\n")