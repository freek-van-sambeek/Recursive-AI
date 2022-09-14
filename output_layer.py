class OutputLayer:
    def __init__(self, nodes, weights=[], biases=[]):
        self.nodes = nodes
        self.weights = weights
        self.biases = biases

    def predict(self):
        prediction = []
        for node in self.nodes:
            prediction.append(node.buffer)
        return prediction

    def lose(self):
        result = 0
        for node in self.nodes:
            result += node.loss
            node.buffer = 0
        return result

    def backpropagate(self, y_):
        for i in range(len(self.nodes)):
            self.nodes[i].backpropagate(y_[i])

    def update(self, learning_rate):
        for node in self.nodes:
            node.update(learning_rate)
