import random as rd


e = 2.718281828459045235360287471352662497757


class Matrix:
    def __init__(self, data):
        self.nrows = len(data)
        self.ncols = len(data[0])
        for i in range(1, self.nrows):
            if len(data[i]) != self.ncols:
                raise ValueError("The lengths of the rows of the data you submitted are not equal.")
        self.data = data

    def dimensions(self):
        return [self.nrows, self.ncols]

    def transpose(self):
        data = []
        for i in range(self.ncols):
            data.append([])
        for i in range(self.ncols):
            for j in range(self.nrows):
                data[i].append(self.data[j][i])
        return Matrix(data)

    def system(self):
        if self.nrows != self.ncols:
            raise ValueError("Number of rows has to be equal to the number of columns to invert a matrix.")
        data = self.data
        identity_matrix_data = identity_matrix(self.nrows).data
        for i in range(self.nrows):
            data[i] += identity_matrix_data[i]
        return Matrix(data)

    def add_sub(self, B, add=True):
        if self.ncols != B.ncols or self.nrows != B.nrows:
            raise ValueError("The number of rows and columns for both matrices must be the same.")
        result = []
        for i in range(self.nrows):
            result.append([])
            for j in range(self.ncols):
                if add:
                    result[i].append(self.data[i][j] + B.data[i][j])
                else:
                    result[i].append(self.data[i][j] - B.data[i][j])
        return Matrix(result)

    def add(self, B):
        return self.add_sub(B)

    def subtract(self, B):
        return self.add_sub(B, False)

    def scale(self, alfa):
        result = []
        for i in range(self.nrows):
            result.append([])
            for j in range(self.ncols):
                result[i].append(alfa * self.data[i][j])
        return Matrix(result)

    def multiply(self, B):
        if self.ncols != B.nrows:
            raise ValueError("The columns of your left matrix are not equal to those of your right matrix.")
        A = self.data
        B_T = B.transpose().data
        result = []
        for i in range(self.nrows):
            result.append([])
            for j in range(B.ncols):
                result[i].append(dotproduct(A[i], B_T[j]))
        return Matrix(result)

    def inverse(self):
        system = self.system()
        data = system.data
        for j in range(self.nrows):
            for i in range(j, self.nrows):
                if data[i][j] != 0:
                    data[i] = row_echelon(data[i], j)
                    if i != j:
                        data[j], data[i] = data[i], data[j]
                    for k in range(self.nrows):
                        if k != j:
                            data[k] = reduce(data[j], data[k], j)
                    break
                if i == self.nrows - 1:
                    raise ValueError("You have a supplied a non-invertible matrix (perfect multicollinearity).")
        inverse = []
        for i in range(self.nrows):
            inverse.append([])
            for j in range(self.nrows):
                inverse[i].append(data[i][self.nrows + j])
        return Matrix(inverse)

    def apply(self, option):
        data = self.data
        result = []
        activation = True
        if option not in {"relu", "sigmoid"}:
            activation = False
        applyfunction = lambda x: (relu(x) if option == "relu" else sigmoid(x)) if activation else (
            x ** 2 if option == "quadratic" else (x ** 2) ** (1 / 2))
        for i in range(self.nrows):
            result.append([])
            for j in range(self.ncols):
                result[i].append(applyfunction(data[i][j]))
        return Matrix(result)

    def gradient(self, option, backpropagated):
        applyfunction = lambda x: relu_prime(x) if option == "relu" else sigmoid_prime(x)
        gradient = []
        new_backpropagated = []
        for j in range(len(self.data[0])):
            gradient.append([])
            new_backpropagated.append(0)
            for i in range(len(self.data)):
                partial_gradient = applyfunction(self.data[i][j]) * backpropagated[i]
                gradient[j].append(partial_gradient)
                new_backpropagated[j] += partial_gradient
        return {"Gradient": Matrix(gradient).transpose(), "Backpropagated": new_backpropagated}

def row_echelon(row, index):
    factor = row[index]
    length = len(row)
    for i in range(length):
        row[i] = row[i] / factor
    return row


def reduce(reducor, reduced, index):
    if reduced[index] == 0:
        return reduced
    factor = reduced[index] / reducor[index]
    length = len(reduced)
    for i in range(length):
        reduced[i] = reduced[i] - factor * reducor[i]
    return reduced


def dotproduct(a, b):
    if len(a) != len(b):
        raise ValueError("The lengths of your vectors must be the same.")
    length = len(a)
    sum = 0
    for i in range(length):
        sum += a[i] * b[i]
    return sum


def identity_matrix(n):
    data = []
    for i in range(n):
        data.append([])
        for j in range(n):
            if i == j:
                data[i].append(1)
            else:
                data[i].append(0)
    return Matrix(data)


def random_matrix(rows, columns):
    data = []
    for i in range(rows):
        data.append([])
        for j in range(columns):
            data[i].append(rd.random() * 2 - 1)
    return Matrix(data)


def sigmoid(x):
    return 1 / (1 + e ** (-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    if x < 0:
        return 0
    else:
        return x


def relu_prime(x):
    return 1 if x > 0 else 0


def loss_prime(x, outcome):
    return 2 * (x + outcome)
