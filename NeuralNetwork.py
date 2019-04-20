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

    def feed_forward(self, inputs):
        self.z_L_minus_one = np.dot(inputs, self.weights_input_to_hidden)
        self.a_L_minus_one = self.sigmoid(self.z_L_minus_one)
        
        self.z_L_one = np.dot(self.a_L_minus_one, self.weights_hidden_to_output)
        self.a_L_one = self.sigmoid(self.z_L_one)

        return self.a_L_one

# Training data
XOR_gate_inputs = np.array( ([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
XOR_gate_outputs = np.array( ([0], [1], [1], [0]), dtype=float)

neural_network = Neural_Network(1)
neural_network_output = neural_network.feed_forward(XOR_gate_inputs)

print("Predicted output:\n" + str(XOR_gate_outputs))
print("Actual output:\n" + str(neural_network_output))