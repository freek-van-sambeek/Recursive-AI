class InputNode:
    def __init__(self, activation, objects_to=[]):
        self.activation = activation
        self.objects_to = objects_to

        self.bool_propagated = False
        self.bool_backpropagated = False
        self.bool_updated = False

    def propagate(self, x):
        result = self.activation.F(x)

        self.bool_propagated = True

        for object in self.objects_to:
            object.propagate(result)

    def backpropagate(self, x):
        self.bool_backpropagated = True

    def update(self, x):
        self.bool_updated = True

    @staticmethod
    def buff(x):
        pass
