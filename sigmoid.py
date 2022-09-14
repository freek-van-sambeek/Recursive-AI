e = 2.718281828459045235360287471352662497757


class Sigmoid:
    # How to skip initialisation
    def __init__(self):
        pass

    def f(self, x):
        return self.F(x) * (1 - self.F(x))

    @staticmethod
    def F(x):
        return 1 / (1 + e ** (- x))


class TanH:
    def __init__(self):
        pass

    def f(self, x):
        return 2 * self.F(x) * (1 - self.F(x))

    @staticmethod
    def F(x):
        return 2 / (1 + e ** (-x)) - 1