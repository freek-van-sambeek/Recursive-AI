class MSE:
    def __init__(self, objects_from=[]):
        self.objects_from = objects_from

    @staticmethod
    def f(x, y):
        # return (2 / n) * (x - y), but for proportionality:
        return 2 * (x - y)

    @staticmethod
    def F(x, y):
        # return (1 / n) * (x - y) ^ 2, but for proportionality:
        return (x - y) ** 2

    def loss(self, x_, y_):
        loss = 0
        for i in range(len(x_)):
            loss += self.F(x_[i], y_[i])
        return loss
