import input_layer
import output_layer
from input_layer import InputLayer
from middle_layer import MiddleLayer
from output_layer import OutputLayer
from input_node import InputNode
from middle_node import MiddleNode
from output_node import OutputNode


class Network:
    def __init__(self, structure, activation, loss_function, learning_rate, activation_output=None, batch=None):
        self.structure = structure
        self.layers = []

        self.activation = activation
        self.activation_output = activation_output
        if not activation_output:
            self.activation_output = self.activation

        self.loss_function = loss_function
        self.loss = 0
        self.learning_rate = learning_rate
        self.batch = batch

        self.create()

        self.min_inputs = []
        self.max_inputs = []
        self.min_outputs = []
        self.max_outputs = []

    def create(self):
        nodes = []
        for j in range(self.structure[0]):
            nodes.append(InputNode(activation=self.activation))
        self.layers.append(InputLayer(nodes))
        for i in range(1, len(self.structure) - 1):
            nodes = []
            for j in range(self.structure[i]):
                nodes.append(MiddleNode(activation=self.activation))
            self.layers.append(MiddleLayer(nodes))
        nodes = []
        for j in range(self.structure[-1]):
            nodes.append(OutputNode(activation=self.activation_output, loss_function=self.loss_function))
        self.layers.append(OutputLayer(nodes))
        self.connect()

    def check_updated(self, iteration, input):
        for layer in self.layers:
            for node in layer.nodes:
                if node.bool_propagated == False or node.bool_backpropagated == False:
                    print(layer, node, node.bool_propagated, node.bool_backpropagated, iteration, input)
                node.bool_propagated = False
                node.bool_backpropagated = False
                node.bool_updated = False

    def connect(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].connect(self.layers[i + 1])

    def propagate(self, inputs):
        self.layers[0].propagate(inputs)

    def predict(self, X):
        self.propagate(X)
        return self.layers[-1].predict()

    def backpropagate(self, outcomes):
        self.layers[-1].backpropagate(outcomes)
        self.lose()

    def lose(self):
        result = self.layers[-1].lose()
        self.loss += result

    def update(self):
        self.layers[-1].update(self.learning_rate)

    def gradient_descent(self, iterations, X, Y):
        batch = self.batch if self.batch else len(X)
        for i in range(iterations):
            for j in range(len(X)):
                self.propagate(X[j])
                self.backpropagate(Y[j])
                # self.check_updated(i, j)
                if (j + 1) % batch == 0:
                    self.update()
                    print(f"Loss is: {self.loss * (len(X) / batch)}")
                    self.loss = 0

    def standardise(self, X, Y):
        result_X = []
        result_Y = []
        self.min_inputs = X[0].copy()
        self.max_inputs = X[0].copy()
        self.min_outputs = X[0].copy()
        self.max_outputs = X[0].copy()

        for j in range(len(X[0])):
            for i in range(len(X)):
                if X[i][j] < self.min_inputs[j]:
                    self.min_inputs[j] = X[i][j]
                elif X[i][j] >= self.max_inputs[j]:
                    self.max_inputs[j] = X[i][j]
        for j in range(len(Y[0])):
            for i in range(len(Y)):
                if Y[i][j] < self.min_outputs[j]:
                    self.min_outputs[j] = Y[i][j]
                elif Y[i][j] >= self.max_outputs[j]:
                    self.max_outputs[j] = Y[i][j]

        for i in range(len(X)):
            result_X.append([])
            for j in range(len(X[0])):
                result_X[i].append(((X[i][j] - self.min_inputs[j]) / (self.max_inputs[j] - self.min_inputs[j])) * 2 - 1)
        for i in range(len(Y)):
            result_Y.append([])
            for j in range(len(Y[0])):
                result_Y[i].append(((Y[i][j] - self.min_outputs[j]) / (self.max_outputs[j] - self.min_outputs[j])) * 2 - 1)

        return [result_X, result_Y]

    def standard(self, matrix, output = False):
        min = self.min_outputs if output else self.min_inputs
        max = self.max_outputs if output else self.max_inputs
        result = []
        for i in range(len(matrix)):
            result.append([])
            for j in range(len(matrix[0])):
                result[i].append((matrix[i][j] - min[j]) / (max[j] - min[j]))
        return result