class MiddleNode:
    def __init__(self, activation, objects_to=[], objects_from=[]):
        self.activation = activation
        self.objects_to = objects_to
        self.objects_from = objects_from
        self.objects_propagated = 0
        self.objects_backpropagated = 0
        self.buffer = 0
        self.propagated = 0

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
            result = self.activation.F(self.buffer)
            for object in self.objects_to:
                object.propagate(result)
            self.objects_propagated = 0
            self.buffer = 0

    def backpropagate(self, x):
        self.buffer += x
        self.objects_backpropagated += 1

        self.bool_backpropagated = True

        if self.objects_backpropagated == len(self.objects_to):
            result = self.activation.f(self.propagated) * self.buffer
            for object in self.objects_from:
                object.backpropagate(result)
            self.objects_backpropagated = 0
            self.buffer = 0

    def update(self, learning_rate):
        self.bool_updated = True

        for object in self.objects_from:
            object.update(learning_rate)
