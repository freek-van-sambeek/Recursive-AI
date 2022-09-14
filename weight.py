class Weight:
    def __init__(self, value, object_to=None, object_from=None):
        self.value = value
        self.object_to = object_to
        self.object_from = object_from
        self.propagated = 0
        self.gradient = 0
        self.buffer = 0

    def propagate(self, x):
        self.buffer += x
        self.propagated = self.buffer
        self.object_to.propagate(self.propagated * self.value)
        self.buffer = 0

    def backpropagate(self, x):
        self.buffer += x * self.propagated
        self.gradient += x * self.propagated
        self.object_from.backpropagate(self.buffer)
        self.buffer = 0

    def update(self, learning_rate):
        self.value -= learning_rate * self.gradient
        self.gradient = 0
        self.object_from.update(learning_rate)
