class Relu:
    def __init__(self):
        pass

    @staticmethod
    def F(x):
        return x if x > 0 else 0

    @staticmethod
    def f(x):
        return 1 if x > 0 else 0
