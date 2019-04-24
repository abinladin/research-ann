import NeuralNetwork as nn
import numpy as np
import os

MAX_NUMBER_OF_NEURONS = 10                 +1
NUMBER_OF_TRAINING_CYCLES = 250            +1
NUMBER_OF_CASES = 4                        +1
NUMBER_OF_ATTEMPTS = 3                     +1

XOR_gate_inputs = np.array( ([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
XOR_gate_outputs = np.array( ([0], [1], [1], [0]), dtype=float)

class DataCollector:

    def __init__(self):
        print("initializing Data Collector\n----------------------------")

        print("creating network_array")
        self.network_array = []
        for i in range (0, MAX_NUMBER_OF_NEURONS - 1):
            self.network_array.append(nn.Neural_Network(i + 1))

        self.case = np.zeros([NUMBER_OF_CASES, NUMBER_OF_ATTEMPTS]) # 2d array of [case][attempt number], storing the final result after x training cycles
    
    
    def attempt_for_all_neuron_sets(self, case_number, attempt_number):
        for number_of_neurons in range(0, MAX_NUMBER_OF_NEURONS - 1):
            print("\t\tNumber of Neurons: ", number_of_neurons + 1)
            for cycle in range(1, NUMBER_OF_TRAINING_CYCLES):
                #print("\t\t\tTraining cycle no. ", cycle)
                self.network_array[number_of_neurons].train(XOR_gate_inputs, XOR_gate_outputs)
            self.case[case_number, attempt_number] = self.network_array[number_of_neurons].feed_forward(XOR_gate_inputs)[case_number]
            print("\t\t Final Value: ",self.case[case_number, attempt_number])


    def add_attempt(self, case_number):
        for attempt_number in range (0, NUMBER_OF_ATTEMPTS - 1):
            print("\tattempt number ", attempt_number + 1)
            self.attempt_for_all_neuron_sets(case_number, attempt_number)
    
    
    def cycle_through_cases(self):
        for case_number in range(0, NUMBER_OF_CASES - 1):
            print("Adding attempts for case ", case_number+1)
            self.add_attempt(case_number)
    
dc = DataCollector()
dc.cycle_through_cases()
