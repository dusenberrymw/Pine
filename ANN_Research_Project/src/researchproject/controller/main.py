'''
Created on Jun 6, 2013

@author: dusenberrymw
'''

from researchproject.model import * 


if __name__ == '__main__':
    #inputs = [[1,2,3]]
    #target_outputs = [[1]]
    inputs = [[0,0],[1,0],[0,1],[1,1]]
    target_outputs = [[0],[1],[1],[0]]
    
    act_func = training.SigmoidActivationFunction()
    network = network.Network(len(inputs[0]), act_func, 3)
    
    print("Before training")
    for i in range(len(inputs)):
        outputs = network.compute_network_output(inputs[i])
        for output in outputs:
            print("Input: %s, Target Output: %s, Actual Output: %s" %(str(inputs[i]), str(target_outputs[i]), str(output)))
    error = network.calculate_error(inputs, target_outputs)   
    print("Error: %s" %(str(error)))
    
    # now train
    trainer = training.Backpropagation()
    trainer.train(inputs, target_outputs, network, act_func, 100000)
    
    print("\nAfter training")
    for i in range(len(inputs)):
        outputs = network.compute_network_output(inputs[i])
        for output in outputs:
            print("Input: %s, Target Output: %s, Actual Output: %s" %(str(inputs[i]), str(target_outputs[i]), str(output)))
    error = network.calculate_error(inputs, target_outputs)   
    print("Error: %s" %(str(error)))
