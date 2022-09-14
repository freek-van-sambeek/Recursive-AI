from matrix import Matrix, random_matrix, loss_prime, sigmoid_prime, relu_prime

class NeuralNetwork:
    def __init__(self, structure, learning_rate, activation="relu", losstype="quadratic"):
        self.structure = structure
        self.learning_rate = learning_rate
        self.activation = activation
        self.losstype = losstype
        self.weights = []
        self.biases = []
        for i in range(1, len(self.structure)):
            self.weights.append(random_matrix(self.structure[i], self.structure[i - 1]))
            self.biases.append(random_matrix(self.structure[i], 1))

    def propagate(self, input):
        result = Matrix([input]).transpose()
        for i in range(len(self.structure) - 1):
            result = self.weights[i].multiply(result)
            result.add(self.biases[i])
            result.apply(self.activation)
        return result

    def loss(self, input, outcome):
        prediction = self.propagate(input)
        outcome = Matrix([outcome]).transpose()
        discrepancy = prediction.subtract(outcome)
        loss_vector = discrepancy.apply(self.losstype)
        data = loss_vector.data
        loss = 0
        for i in range(len(data)):
            loss += data[i][0]
        return loss / len(data)

    def mse(self, inputs, outcomes):
        loss = 0
        for i in range(len(inputs)):
            loss += self.loss(inputs[i], outcomes[i])
        return loss / len(inputs)

    def backpropagate(self, input, outcome):
        prediction = self.propagate(input).transpose().data[0]
        backpropagated = []
        updates = []
        loss_gradient = []
        applyfunction = lambda x: relu_prime(x) if self.activation == "relu" else sigmoid_prime(x)
        for j in range(len(self.weights[-1].data[0])):
            loss_gradient.append([])
            backpropagated.append(0)
            for i in range(len(self.weights[-1].data)):
                partial_gradient = loss_prime(prediction[i], outcome[i]) * applyfunction(self.weights[-1].data[i][j])
                loss_gradient[j].append(partial_gradient)
                backpropagated[j] += partial_gradient
        for j in range(len(self.biases[-1].data)):
            partial_gradient = loss_prime(prediction[j], outcome[j]) * applyfunction(self.biases[-1].data[j][0])
            self.biases[-1].data[j][0] -= self.learning_rate * partial_gradient

        updates.append(Matrix(loss_gradient).transpose())
        for i in range(1, len(self.weights)):
            for j in range(len(self.biases[-(1 + i)].data)):
                partial_gradient = backpropagated[j] * applyfunction(self.biases[-(1 + i)].data[j][0])
                self.biases[-(1 + i)].data[j][0] -= self.learning_rate * partial_gradient
            update = self.weights[-(1 + i)].gradient(self.activation, backpropagated)
            updates.append(update["Gradient"])
            backpropagated = update["Backpropagated"]
        return updates
        # If stochastic gradient descent:
        #   self.update_parameters(updates, self.learning_rate)

    def update_parameters(self, updates, learning_rate):
        for i in range(len(updates)):
            self.weights[-(1 + i)] = self.weights[-(1 + i)].subtract(updates[i].scale(learning_rate))

    def descend_gradient(self, inputs, outcomes):
        updates = self.backpropagate(inputs[0], outcomes[0])
        for i in range(1, len(outcomes)):
            update = self.backpropagate(inputs[i], outcomes[i])
            for j in range(len(updates)):
                updates[j].add(update[j])
        self.update_parameters(updates, self.learning_rate)

    def gradient_descent(self, inputs, outcomes, epochs):
        for i in range(epochs):
            self.descend_gradient(inputs, outcomes)
            # print(self.mse(inputs, outcomes))
            # for j in range(len(self.weights)):
            #     print(self.weights[j].data)
            #     print(self.biases[j].data)
            print(f'Epoch: {i + 1}')
        print('Finished gradient descent.')

def normalise(array):
    max = array[0][0]
    min = array[0][0]
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i][j] > max:
                max = array[i][j]
                break
            if array[i][j] < min:
                min = array[i][j]
    for i in range(len(array)):
        for j in range(len(array[0])):
            array[i][j] = (array[i][j] - max + min) / (max - min)