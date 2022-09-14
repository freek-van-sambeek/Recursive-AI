import matrix as mx
from middle_layer import Layer


class Connection(Layer):
    def __init__(self, input_size, output_size):
        self.weights = mx.random_matrix(output_size, input_size)
        self.biases = mx.random_matrix(output_size, 1)

    def propagate(self, input_data):
        self.input = input_data
        self.output = self.weights.multiply(input_data).add(self.biases)
        return self.output

    def backpropagate(self, output_error, learning_rate):
        input_error = mx.mult(output_error, self.weights.T)
        weights_error = mx.mult(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
