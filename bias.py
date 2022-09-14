class Bias:
    def __init__(self, value, object_to=None):
        self.value = value
        self.object_to = object_to
        self.gradient = 0

    def backpropagate(self, x):
        self.gradient += x * 1

    def update(self, learning_rate):
        self.value -= learning_rate * self.gradient
        self.gradient = 0
