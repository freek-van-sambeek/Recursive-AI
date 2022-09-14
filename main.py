from network import Network
from sigmoid import Sigmoid, TanH
from relu import Relu
from linear import Linear
from mse import MSE
import csv
import random as rd
import timeit

# dataset = []
# with open("attend.csv", newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         dataset.append(row)


def generate_input():
    return [rd.random() * 20 - 10, rd.random() * 20 - 10]


def generate_inputs(size):
    result = []
    for i in range(size):
        result.append(generate_input())
    return result


def generate_outcome(input):
    return [input[0] ** 2 - 3 * input[1] + 2]


def generate_outcomes(inputs):
    result = []
    for i in range(len(inputs)):
        result.append(generate_outcome(inputs[i]))
    return result


sigmoid = Sigmoid()
tanh = TanH()
relu = Relu()
linear = Linear()

structure = [2, 2, 1]
mse = MSE()
network = Network(structure=structure, activation=sigmoid, activation_output=tanh, loss_function=mse, learning_rate=.02)

