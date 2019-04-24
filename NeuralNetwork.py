import numpy as np
import os
class Neural_Network:

    def __init__(self, hidden_layer_size):
        # Layers
        self.input_layer_size = 2
        self.hidden_layer_size = hidden_layer_size + 1
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
    
    # TODO: refactor this shit so that it makes a modicrum of sense
    def backpropogate(self, X, expected_output, predicted_output):
        self.output_error = (expected_output - predicted_output)
        self.output_error_delta = self.output_error * self.d_sigmoid(predicted_output)

        self.weights_error = self.output_error_delta.dot(self.weights_hidden_to_output.T)
        self.weights_error_delta = self.weights_error * self.d_sigmoid(self.a_L_minus_one)

        self.weights_input_to_hidden += X.T.dot(self.weights_error_delta)
        self.weights_hidden_to_output += self.a_L_minus_one.T.dot(self.output_error_delta)
    
    def train(self, inputs, expected_output):
        predicted_output = self.feed_forward(expected_output)
        self.backpropogate(inputs, expected_output, predicted_output)
