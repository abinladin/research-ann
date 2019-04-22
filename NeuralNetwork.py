import numpy as np

class Neural_Network:

    def __init__(self, hidden_layer_size):
        # Layers
        self.input_layer_size = 2
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = 1

        # Weights
        self.weights_input_to_hidden = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.weights_hidden_to_output = np.random.randn(self.hidden_layer_size, self.output_layer_size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def d_sigmoid(self, z):
        return z * (1 - z)

    def feed_forward(self, inputs):
        self.z_L_minus_one = np.dot(inputs, self.weights_input_to_hidden)
        self.a_L_minus_one = self.sigmoid(self.z_L_minus_one)
        
        self.z_L_one = np.dot(self.a_L_minus_one, self.weights_hidden_to_output)
        self.a_L_one = self.sigmoid(self.z_L_one)

        return self.a_L_one
    
    def backpropogate(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.d_sigmoid(o)

        self.z2_error = self.o_delta.dot(self.weights_hidden_to_output.T)
        self.z2_delta = self.z2_error * self.d_sigmoid(self.a_L_minus_one)

        self.weights_input_to_hidden += X.T.dot(self.z2_delta)
        self.weights_hidden_to_output += self.a_L_minus_one.T.dot(self.o_delta)
    
    def train(self, X, y):
        o = self.feed_forward(X)
        self.backpropogate(X, y, o)

    def predict(self):
        print("Predicted data based on weights:")
        print("Input: \n" + str(XOR_gate_inputs))
        print("output: \n" + str(self.feed_forward(XOR_gate_inputs)))

# Training data
XOR_gate_inputs = np.array( ([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
XOR_gate_outputs = np.array( ([0], [1], [1], [0]), dtype=float)

neural_network = Neural_Network(20)
neural_network_output = neural_network.feed_forward(XOR_gate_inputs)

for i in range(500):
    neural_network.train(XOR_gate_inputs, XOR_gate_outputs)

neural_network.predict()