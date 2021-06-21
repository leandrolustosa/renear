import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(size=20, print_dataset=False):
    rng = np.random.RandomState(42)

    X = rng.uniform(size=(size, 2))
    y = np.array(X[:, 0] > X[:, 1], dtype=int)*2-1

    if print_dataset==True:
        print(X)
        print(y)

    return X, y


def plot_dataset(X, y):
    for k in set(y):
        plt.plot(X[:, 0][y == k], X[:, 1][y == k], "o", alpha=0.3)


def plot_vetor(vetor, color="k"):
    origem = [0, 0]
    plt.arrow(*origem, *vetor, color=color)
    plt.show()


def perpendicular(hiperplano):
    perp = np.array(hiperplano[::-1])
    perp[0] *= -1

    return perp


def step_func(y):
    return (y >= 0)*2 - 1

def accuracy(y, y_pred):
    accuracy = sum(y_pred == y) / len(y)
    print(f"Accuracy={accuracy}")

class CostPerceptron():
    @staticmethod
    def error(y, y_pred):
        return y - y_pred
    @staticmethod
    def cost(y, y_pred):
        return np.sum(CostPerceptron.error(y, y_pred)**2)
    @staticmethod
    def gradient(y, y_pred, X):
        return np.matmul(CostPerceptron.error(y, y_pred), X)

class HingeLoss():
    @staticmethod
    def error(y, y_pred):
        return np.multiply(y, y_pred)
    @staticmethod
    def cost(y, y_pred):
        return np.sum(1 - HingeLoss.error(y, y_pred))
    @staticmethod
    def gradient(y, y_pred, X):
        marginals = HingeLoss.error(y, y_pred) < 1
        return np.matmul(y[marginals], X[marginals])

class Adaline():
    def __init__(self):
        self.pre_activated = True

    @staticmethod
    def error(y, y_pred):
        return y - y_pred
    @staticmethod
    def cost(y, y_pred):
        return np.sum((1 - Adaline.error(y, y_pred))**2)
    @staticmethod
    def gradient(y, y_pred, X):
        return np.matmul(Adaline.error(y, y_pred), X)

class Perceptron(object):

    def __init__(self, activation=step_func, cf=CostPerceptron()):
        self.weights = None
        self.bias = None
        self.activation = step_func
        self.cf = cf

    def fit(self, X, y, lr=1e-1, epochs=5):
        # inicializando com valores aleatórios entre -1 e 1
        self.weights = np.random.uniform(1, -1, size=2)
        self.bias = 2 * np.random.random() - 1

        for epoch in range(1, epochs + 1):
            if hasattr(self.cf, 'pre_activated') and self.cf.pre_activated:
                y_pred = self.pre_activate(X)
            else:
                y_pred = self.predict(X)
            loss = self.cf.error(y, y_pred)
            cost = self.cf.cost(y, y_pred)

            self.weights = self.weights + lr * self.cf.gradient(y, y_pred, X)
            self.bias = self.bias + lr * loss

            print(f"epoch={epoch} - cost={cost}")

            if cost == 0:
                break

    def pre_activate(self, X):
        return np.matmul(X, self.weights)

    def predict(self, X):
        output = self.pre_activate(X)
        y_pred = self.activation(output)
        return y_pred

if __name__ == "__main__":
    X, y = generate_dataset()

    plot_dataset(X, y)
    hiperplano = [1, -1]
    vetor = perpendicular(hiperplano)    
    #plot_vetor(vetor)

    cfs = [CostPerceptron(), HingeLoss(), Adaline()]
    X_test, y_test = generate_dataset(1000)

    for cf in cfs:
        perceptron = Perceptron(cf=cf)

        perceptron.fit(X, y, epochs=100)
        y_pred = perceptron.predict(X_test)
        accuracy(y_test, y_pred)

        # plot_dataset(X_test, y_test)
        # plot_vetor(perpendicular(hiperplano))