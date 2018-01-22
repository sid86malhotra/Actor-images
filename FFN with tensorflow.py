import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import shuffle


#Read in the training data
Train_data = pd.read_csv("Training.csv").as_matrix()

X = Train_data[:, 2:] / 255
Y = Train_data[:, 1]

#Encode the texual values into a numeric format.
lb = LabelEncoder()
Y = lb.fit_transform(Y)

#Now we will need to onehotencode the values
Ytemp = Y.reshape(-1, 1) #We need to reshape the as OneHotEncoder only takes in 2D values
onc = OneHotEncoder()
T = onc.fit_transform(Ytemp).toarray()

#Now we will divide our data into testing and training sets
X, Y, T = shuffle(X, Y, T) # Shuffle the data first

Xtrain, Ytrain, Ttrain = X[:18000], Y[:18000], T[:18000]
Xtest, Ytest, Ttest = X[18000:], Y[18000:], T[18000:]

#########################################################
#   tensorflow model prep
#########################################################

#Setup the layers
N, D = X.shape
K = len(set(Y))
L1 = 1500
L2 = 600


def initialize_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

def forward_1layers(X, W1, b1, W2, b2):
    Z = tf.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2


def initialize_weights_for_1_layers(N, D, L1, K):
    W1 = initialize_weights([D, L1])
    b1 = initialize_weights([L1])
    W2 = initialize_weights([L1, K])
    b2 = initialize_weights([K])
    return W1, b1, W2, b2

def forward_2layers(X, W1, b1, W2, b2, W3, b3):
    # Z1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    # Z2 = tf.sigmoid(tf.matmul(Z1, W2) + b2)
    Z1 = tf.tanh(tf.matmul(X, W1) + b1)
    Z2 = tf.tanh(tf.matmul(Z1, W2) + b2)
    return tf.matmul(Z2, W3) + b3


def initialize_weights_for_2_layers(N, D, L1, L2, K):
    W1 = initialize_weights([D, L1])
    b1 = initialize_weights([L1])
    W2 = initialize_weights([L1, L2])
    b2 = initialize_weights([L2])
    W3 = initialize_weights([L2, K])
    b3 = initialize_weights([K])
    return W1, b1, W2, b2, W3, b3

#Setup the placeholders for X and Y
tfX = tf.placeholder(tf.float32, shape = (None, D))
tfY = tf.placeholder(tf.float32, shape = (None, K))

# W1, b1, W2, b2 = initialize_weights_for_1_layers(N, D, L1, K)
W1, b1, W2, b2, W3, b3 = initialize_weights_for_2_layers(N, D, L1, L2, K)


#Predict function
# logits = forward_1layers(tfX, W1, b1, W2, b2)
logits = forward_2layers(tfX, W1, b1, W2, b2, W3, b3)

#Define the cost function
Cross_entropy_costs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tfY, logits = logits))

#Training function
train_op = tf.train.GradientDescentOptimizer(10e-2).minimize(Cross_entropy_costs)

predict_op = tf.argmax(logits, axis = 1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

costs_train = []
costs_test = []

for epoch in range(500):
    _, loss_value_train, predtrain = sess.run([train_op, Cross_entropy_costs, predict_op],
                        feed_dict = {tfX: Xtrain, tfY: Ttrain})

    loss_value_test, predtest = sess.run([Cross_entropy_costs, predict_op],
                        feed_dict = {tfX: Xtest, tfY: Ttest})
    costs_train.append(loss_value_train)
    costs_test.append(loss_value_test)

    if epoch % 2 == 0:
        acc_train = np.mean(predtrain == Ytrain)
        acc_test = np.mean(predtest == Ytest)
        print("Epoch:", epoch, "Accuracy:", acc_train, "Costs:", loss_value_train,
                        "Acc Test:", acc_test, "Costs Test:", loss_value_test)

_ = plt.plot(costs_train, label = "Train")
_ = plt.plot(costs_test, label = "Test")
legend = plt.legend(loc = 1)
plt.show()

#We will now read the Test dataset
data = pd.read_csv("Test.csv").as_matrix()
XValid = (data[:,1:] / 255).astype(np.float32)
ID = data[:,0]

TestResult = sess.run(predict_op, feed_dict = {tfX : XValid})

def GetClass(Y):

    N = Y.shape[0]
    T = []
    for row in range(N):
        if Y[row] == 0:
            T.append("OLD")
        elif Y[row] == 1:
            T.append("MIDDLE")
        else:
            T.append("YOUNG")
    return T

Submission = pd.DataFrame({"Class" : np.array(TestResult),
                    "ID" : ID})

Submission.to_csv("submit.csv", index = False)
