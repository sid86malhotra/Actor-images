import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def softmax(input):
    return np.exp(input) / np.exp(input).sum(axis = 1, keepdims = True)

# def cross_entropy_loss(Y, T):
#     N = len(T)
#     return -np.log(Y[np.arange(N), T.astype(np.int32)]).mean()

def cross_entropy_loss(Y, T):
    return -(T * np.log(Y)).mean()


def calcualte_accuracy(output, Y):
    pY = np.argmax(output, axis = 1)
    return np.mean(pY == Y)

#Now we convert the target values to numeric
def label_onehot_encode(Y):

    N = len(Y)

    output = []
    T = np.zeros((N, len(set(Y))))

    for row in range(N):
        if Y[row] == "OLD":
            T[row, 2] = 1
            output.append(2)
        elif Y[row] == "MIDDLE":
            T[row, 1] = 1
            output.append(1)
        else:
            T[row, 0] = 1
            output.append(0)

    outputNUMPY = np.array(output)
    return T, outputNUMPY

#Convert numerical classes back to alphanumeric
def GetClass(Y):

    N = len(Y)
    T = []
    for row in range(N):
        if Y[row] == 0:
            T.append("OLD")
        elif Y[row] == 1:
            T.append("MIDDLE")
        else:
            T.append("YOUNG")

    return T

class FFNNUMPY(object):

    def __init__(self, M):
        self.M = M

    def predict(self, X, get_weights = False):
        if get_weights:
            print("Check if this is running")
            self.W1 = pd.read_csv("W1.csv").as_matrix()
            self.W2 = pd.read_csv("W2.csv").as_matrix()
            self.b1 = pd.read_csv("b1.csv").as_matrix()
            self.b2 = pd.read_csv("b2.csv").as_matrix()
            print(self.W1.shape, self.W2.shape, self.b1.shape, self.b2.shape)
        hidden = np.tanh(X.dot(self.W1) + self.b1)
        output = softmax(hidden.dot(self.W2) + self.b2)
        return hidden, output

    def initiate_weights(self, N, D, K):
        W1 = np.random.randn(D, self.M) / np.sqrt(N + D)
        b1 = np.zeros(self.M)
        W2 = np.random.randn(self.M, K) / np.sqrt(N + D)
        b2 = np.zeros(K)
        return W1, b1, W2, b2

    def save_weights_func(self):

        df = pd.DataFrame(self.W1)
        df.to_csv("W1.csv", index = False)
        df = pd.DataFrame(self.W2)
        df.to_csv("W2.csv", index = False)
        df = pd.DataFrame(self.b1)
        df.to_csv("b1.csv", index = False)
        df = pd.DataFrame(self.b2)
        df.to_csv("b2.csv", index = False)

    def score_function(self, X, Y, get_weights = False):
        _, output = self.predict(X, get_weights)
        print(output)
        return calcualte_accuracy(output, Y)

    def fit(self, X, Y, T, learning_rate = 10e-7, reg = 10e-6, epochs = 20, batch_size = 500, show_fig = True, save_weights = True):

        N, D = X.shape
        K = len(set(Y))

        self.W1, self.b1, self.W2, self.b2 = self.initiate_weights(N, D, K)

        num_batches = np.round(N / batch_size).astype(np.int32)

        costs = []

        for epoch in range(epochs):
            X, Y, T = shuffle(X, Y, T)

            for batch in range(num_batches):

                Xbatch = X[batch * batch_size: batch_size * (batch + 1)]
                Tbatch = T[batch * batch_size: batch_size * (batch + 1)]
                Ybatch = Y[batch * batch_size: batch_size * (batch + 1)]

                hidden, output = self.predict(Xbatch)

                pY_T = (output - Tbatch)

                self.W2 -= learning_rate * (hidden.T.dot(pY_T) + reg * self.W2)
                self.b2 -= learning_rate * (pY_T.sum(axis = 0) + reg * self.b2)

                dZ = pY_T.dot(self.W2.T) * hidden * (1 - hidden)

                self.W1 -= learning_rate * (Xbatch.T.dot(dZ) + reg * self.W1)
                self.b1 -= learning_rate * (dZ.sum(axis = 0) + reg * self.b1)
                c = cross_entropy_loss(output, Tbatch)
                costs.append(c)
                a = calcualte_accuracy(output, Ybatch)
                print("Epoch", epoch, "Batch", batch, "Costs", c, "Accuracy", a)



        if show_fig:
            _ = plt.plot(costs)
            plt.show()

        self.save_weights_func()



def main():

    data = pd.read_csv("Training.csv").as_matrix()

    X = (data[:, 2:] / 255).astype(np.float32)
    Y = data[:,1]

    T, Y = label_onehot_encode(Y)
    print(X.shape)

    #We would need to setup the training and testing samples from the data
    X, Y, T = shuffle(X, Y, T)

    Xtrain, Ytrain, Ttrain = X[:18000], Y[:18000], T[:18000]
    Xtest, Ytest, Ttest = X[18000:], Y[18000:], T[18000:]

    model = FFNNUMPY(M = 1400)

    model.fit(Xtrain, Ytrain, Ttrain, epochs = 600, learning_rate = 10e-7, batch_size = 1000)
    print(model.score_function(Xtest, Ytest))


    #We will now read the Test dataset
    data = pd.read_csv("Test.csv").as_matrix()
    X = (data[:,1:] / 255).astype(np.float32)
    ID = data[:,0]

    _, TestResult = model.predict(X)
    pY = np.argmax(TestResult, axis = 1)


    Submission = pd.DataFrame({"Class" : np.array(GetClass(pY)),
                        "ID" : ID})

    Submission.to_csv("submit.csv", index = False)

if __name__ == "__main__":
    main()
