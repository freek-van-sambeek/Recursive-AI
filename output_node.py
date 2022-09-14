class OutputNode:
    def __init__(self, activation, loss_function, objects_from=[]):
        self.activation = activation
        self.objects_from = objects_from
        self.objects_propagated = 0
        self.loss_function = loss_function
        self.propagated = 0
        self.buffer = 0
        self.loss = 0

        self.bool_propagated = False
        self.bool_backpropagated = False
        self.bool_updated = False

    def propagate(self, x):
        self.buffer += x
        self.objects_propagated += 1

        self.bool_propagated = True

        if self.objects_propagated == len(self.objects_from) - 1:
            self.buffer += self.objects_from[-1].value
            self.propagated = self.buffer
            self.buffer = self.activation.F(self.buffer)
            self.objects_propagated = 0

    def backpropagate(self, y):
        self.bool_backpropagated = True

        self.loss = self.loss_function.F(self.buffer, y)
        result = self.activation.f(self.propagated) * self.loss_function.f(self.buffer, y)
        for object in self.objects_from:
            object.backpropagate(result)
        self.buffer = 0

    def update(self, learning_rate):
        self.bool_updated = True

        for object in self.objects_from:
            object.update(learning_rate)
