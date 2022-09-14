from weight import Weight
from bias import Bias
import random as rd


class InputLayer:
    def __init__(self, nodes):
        self.nodes = nodes

    def propagate(self, inputs):
        for i in range(len(self.nodes)):
            self.nodes[i].propagate(inputs[i])

    def connect(self, layer):
        for node_own in self.nodes:
            for node_other in layer.nodes:
                weight = Weight(value=rd.random() * 2 - 1, object_from=node_own, object_to=node_other)
                layer.weights.append(weight)
                node_own.objects_to.append(weight)
                node_other.objects_from.append(weight)
        for node_other in layer.nodes:
            bias = Bias(value=rd.random() * 2 - 1, object_to=node_other)
            layer.biases.append(bias)
            node_other.objects_from.append(bias)
