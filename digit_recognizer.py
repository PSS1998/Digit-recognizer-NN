import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    B = np.copy(x)
    # B = np.round(B, 10)
    return sigmoid(B) * (1.0 - sigmoid(B))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return (1 - (np.tanh(x) ** 2))


def loss_function(output_train, y_train, weights, lmbda):
    return (1/2)*((y_train - output_train)**2)

def loss_function_derivative(output_train, y_train):
    return (output_train - y_train)


class NeuralNetwork:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.input_train    = x_train
        self.input_test     = x_test
        self.input          = x_test
        self.weights1       = np.random.rand(256, self.input_train.shape[1]) 
        self.bias1          = np.random.rand(256)
        self.weights2       = np.random.rand(10, 256)        
        self.bias2          = np.random.rand(10)
        self.y_train        = y_train
        self.y_test         = y_test
        self.output_train   = np.zeros([self.y_train.shape[0],10])
        self.output_test    = np.zeros([self.y_test.shape[0],10])
        self.output         = np.zeros([1,10])
        self.result_train   = np.zeros(self.y_train.shape)
        self.result_test    = np.zeros(self.y_test.shape)
        self.result         = np.zeros(1)
        self.alpha          = 0.05
        self.lmbda          = 0.05

    def calculate_result_train(self):
        self.result_train = self.output_train.argmax(axis=0)

    def calculate_result_test(self):
        self.result_test = self.output_test.argmax(axis=0)

    def calculate_result(self):
        self.result = self.output.argmax(axis=0)

    def feedforward_train(self, i):
        self.layer1 = tanh(np.dot(self.weights1, self.input_train[i]) + self.bias1)
        self.output_train = sigmoid(np.dot(self.weights2, self.layer1) + self.bias2)
        self.calculate_result_train()

    def feedforward_test(self, i):
        self.layer1 = tanh(np.dot(self.weights1, self.input_test[i]) + self.bias1)
        self.output_test = sigmoid(np.dot(self.weights2, self.layer1) + self.bias2)
        self.calculate_result_test()

    def feedforward(self):
        self.layer1 = tanh(np.dot(self.weights1, self.input) + self.bias1)
        self.output = sigmoid(np.dot(self.weights2, self.layer1) + self.bias2)
        self.calculate_result()

    def backpropagation(self, i):
        hidden_output = self.layer1
        inputlayer_output = self.input_train[i]

        output_error = (loss_function_derivative(self.output_train, self.y_train[i]) * sigmoid_derivative(self.output_train))
        hidden_error = (np.dot(loss_function_derivative(self.output_train, self.y_train[i]) * sigmoid_derivative(self.output_train), self.weights2) * tanh_derivative(self.layer1))

        d_weights2 = np.empty([output_error.shape[0], hidden_output.shape[0]])
        for i in range(output_error.shape[0]):
            d_weights2[i] = (output_error[i]*hidden_output)
        d_bias2 = output_error

        d_weights1 = np.empty([hidden_error.shape[0], inputlayer_output.shape[0]])
        for i in range(hidden_error.shape[0]):
            d_weights1[i] = (hidden_error[i]*inputlayer_output)
        d_bias1 = hidden_error

        self.weights1 += -(self.alpha * d_weights1)
        self.weights2 += -(self.alpha * d_weights2)

        self.bias1 += -(self.alpha * d_bias1)
        self.bias2 += -(self.alpha * d_bias2)


    def train(self):
        self.weights1 = self.weights1-0.5
        self.weights1 = self.weights1*2
        self.weights2 = self.weights2-0.5
        self.weights2 = self.weights2*2
        for i in range(60000):
            self.feedforward_train(i)
            self.backpropagation(i)

    def test(self):
        wrong = 0
        for i in range(10000):
            self.feedforward_test(i)
            if(not self.y_test[i][self.result_test]):
                wrong += 1
        print("accurecy : ", (10000 - wrong)/100)

    def predict(self, x):
        self.input = x
        self.feedforward()
        self.calculate_result()
        print("prediction : ",self.result)


def loadMNIST( prefix, folder ):
    intType_bigendian = np.dtype('int32').newbyteorder('>')

    nMetaDataBytes = 4 * intType_bigendian.itemsize

    data = np.fromfile( folder + "/" + prefix + '-images-idx3-ubyte', dtype = 'ubyte' )
    magicNumber, nImages, nRows, nCols = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType_bigendian )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, nRows*nCols ] )

    nMetaLabelBytes = 2 * intType_bigendian.itemsize

    labels = np.fromfile( folder + "/" + prefix + '-labels-idx1-ubyte', dtype = 'ubyte' )[nMetaLabelBytes:]

    return data, labels


def vectorized_result(label):
    e = np.zeros(10)
    e[label] = 1.0
    return e


def normalize(d):
    return d/255




trainingImages, trainingLabels = loadMNIST( "train", "./datasets/" )
testImages, testLabels = loadMNIST( "t10k", "./datasets/" )


trainingLabels2 = np.empty([60000, 10])
for i in range(60000):
    trainingLabels2[i] = vectorized_result(trainingLabels[i])
trainingLabels = trainingLabels2

testLabels2 = np.empty([10000, 10])
for i in range(10000):
    testLabels2[i] = vectorized_result(testLabels[i])
testLabels = testLabels2


trainingImages = normalize(trainingImages)
testImages = normalize(testImages)

                                        
X_train = trainingImages
y_train = trainingLabels
X_test = testImages
y_test = testLabels


nn = NeuralNetwork(X_train, y_train, X_test, y_test)

nn.train()
nn.test()
nn.predict(X_test[4521])