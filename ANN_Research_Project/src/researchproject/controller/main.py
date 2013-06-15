'''
Created on Jun 6, 2013

@author: dusenberrymw
'''

import sys,os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from researchproject.model.network import Network
from researchproject.model import training
from researchproject.model import data
import time
import cProfile
import csv

    
def main():
    # this will be used for timing the code
    start_time = time.clock()
    
    # Set up the inputs and target outputs
    inputs = []
    target_outputs = []
    temp_list = []
    row_length = 0
    with open(os.path.join(os.path.dirname(__file__), 
                           '../model/data/CT_Data_Edited.csv'), 
                           newline='') as data_file:
            data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
            i = 0
            for row in data_reader:
                row_length = len(row)
                temp_list = [int(x) for x in row]
                inputs.append(temp_list[:row_length-1])
                target_outputs.append(temp_list[row_length-1:row_length])
    
            data_file.close()
#     halfway_point = len(inputs)//2 # the // operator returns an integer
#     training_inputs = inputs[:halfway_point]
#     training_target_outputs = target_outputs[:halfway_point]
#     testing_inputs = inputs[halfway_point:]
#     testing_target_outputs = target_outputs[halfway_point:]
    
    training_inputs = inputs
    training_target_outputs = target_outputs
    testing_inputs = training_inputs
    testing_target_outputs = training_target_outputs
   
#     training_inputs = [[0,0],[1,0],[0,1],[1,1]]
#     training_target_outputs = [[0],[1],[1],[0]]
#     testing_inputs = training_inputs
#     testing_target_outputs = training_target_outputs
            
    
    # Create the network
    act_func = training.SigmoidActivationFunction()
#     num_hidden_layers = 6
#     network = Network(len(training_inputs[0]), act_func, num_hidden_layers)
    network = Network(len(training_inputs[0]), act_func)
    print(len(network.hidden_layer.neurons))
    
    # Test network prior to training
    print("Before training")
    for i in range(len(testing_inputs)):
        outputs = network.compute_network_output(testing_inputs[i])
        for output in outputs:
            print("Input: %s, Target Output: %s, Actual Output: %s" 
                  %(str(testing_inputs[i]), str(testing_target_outputs[i]), str(output)))
    error = network.calculate_error(testing_inputs, testing_target_outputs)   
    print("Error: %s \n" %(str(error)))
    
    # Train the network
    min_error = 0.00001
    iterations = 1000
    print("Will train for %s iterations \n" % str(iterations))
    trainer = training.Backpropagation()
    trainer.train(training_inputs, training_target_outputs, network, act_func, iterations, min_error)
    
    # Test the network after training
    print("After training")
    for i in range(len(testing_inputs)):
        outputs = network.compute_network_output(testing_inputs[i])
        for output in outputs:
            print("Input: %s, Target Output: %s, Actual Output: %s" 
                  %(str(testing_inputs[i]), str(testing_target_outputs[i]), str(output)))
    error = network.calculate_error(testing_inputs, testing_target_outputs)   
    print("Error: %s" %(str(error)))
    print("Trained for %s iterations" % (str(trainer.iterations)))
    
    print("Code took %s seconds to run\n" % (str(time.clock() - start_time)))
    
    
if __name__ == '__main__':
    cProfile.run('main()')
#     main()

#      neuron_weights = [1,1,1]
#     deltas = [2,2,2]
#     momentums = [3,3,3]
# 
#     new_weights  = [sum(i) for i in zip(neuron_weights, deltas, momentums)]
#      
#      new_weights = neuron_weights
#      new_weights[0] = 9
#      print(neuron_weights)
#      print(new_weights)

    
    