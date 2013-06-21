'''
Created on Jun 6, 2013

@author: dusenberrymw
'''

import sys,os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from researchproject.model.network import Network
from researchproject.model import network as network_module
from researchproject.model import training
from researchproject.model import data as data_module
import time
import copy
from multiprocessing import cpu_count
import cProfile

# constants for the different projects
XOR_PROJECT = 1
IRIS_PROJECT = 2
CT_PROJECT = 3
LETTER_RECOG_PROJECT = 4

def main(project):
    """Starting point for this program
    
    Will build a default list of parameters, get the data for the specified
    project, then build and train networks in parallel, trying different
    configurations
    
    """
    # build the params for the project
    params = build_project_params(project)
    
    # create the network
    network = Network(len(params['data'].training_inputs[0]), 
                      params['act_func'],
                      params['num_hidden_layer1_neurons'], 
                      params['two_hidden_layers'],
                      params['num_hidden_layer2_neurons'],
                      len(params['data'].training_target_outputs[0]))
#     # Test the network
#     network2 = copy.deepcopy(network)
#     test_run(network2, params)
    
    print('\nNumber of hidden neurons: {0}'.format(len(network.layers[0].neurons)))
    print('Number of output neurons: {0}'.format(len(network.layers[-1].neurons)))
    print('Number of total layers: {0}'.format(len(network.layers)))
    
    # test the network
    print("\nBefore training")
    network_module.print_network_error(network, params['data'])
    
    # train the network
    trainer = training.Backpropagation()
    i = 0
    error = network.calculate_error(params['data'].training_inputs, 
                                    params['data'].training_target_outputs)
    while (i<10) & (error > params['min_error']):
        training.parallel_training(network, trainer, params)
        
        # check the new error on the master network
        print("\nMaster Network:")
        network_module.print_network_error(network, params['data'])
        error = network.calculate_error(params['data'].training_inputs, 
                                    params['data'].training_target_outputs)
        i +=1
        print('{0}\t{1}'.format(params['learning_rate'], params['momentum_coef']))
        
        # change the learning rate and momentum coef
        params['learning_rate'] += params['learning_rate']*params['learning_rate_change']
        params['momentum_coef'] += params['momentum_coef']*params['momentum_coef_change']
    
#     network_module.print_network_outputs(network, params['data'])


def build_project_params(project):
    """Create the default parameters for this project"""
    # create a dictionary to store all params
    params = {}
    
    # Set up default parameters
    params['act_func'] = training.TanhActivationFunction()
    params['num_hidden_layer1_neurons'] = None
    params['num_hidden_layer2_neurons'] = None
    params['two_hidden_layers'] = False
    params['learning_rate'] = 0.1
    params['momentum_coef'] = 0.5
    params['learning_rate_change'] = 0
    params['momentum_coef_change'] = 0
    params['min_error'] = 0.0001
    params['iterations'] = 1000
    params['num_processes'] = cpu_count()
    
    # Get the project's data and any overrides to the defaults
    if project == XOR_PROJECT:
        params['data'] = data_module.xor_data()
        params['act_func'] = training.SigmoidActivationFunction()
        params['num_hidden_layer1_neurons'] = 3
        params['learning_rate'] = 0.7
        params['momentum_coef'] = 0.9
        params['iterations'] = 10000
        params['num_processes'] = 1
    elif project == IRIS_PROJECT:
        params['act_func'] = training.TanhActivationFunction()
        params['data'] = data_module.iris_data()
        params['learning_rate'] = 0.01 #0.2
        params['momentum_coef'] = 0.1 #0.4
        params['learning_rate_change'] = 0.0 #0.2
        params['momentum_coef_change'] = 0.0 #0.2
        params['iterations'] = 1000
    elif project == LETTER_RECOG_PROJECT:
        params['data'] = data_module.letter_recognition_data()
        params['learning_rate'] = 0.1
        params['momentum_coef'] = 0.5 
        params['learning_rate_change'] = 0.01 #0.2
        params['momentum_coef_change'] = 0.0 #0.2
        params['iterations'] = 500
    elif project == CT_PROJECT:
        params['data'] = data_module.ct_data()
    
    return params


def test_run(network, params):
    """Test method for the system
    
    This creates a network, tests it, trains it, and retests it
    
    """
    params['iterations'] = 10000
    # Create the network
#     network = Network(len(params['data'].training_inputs[0]), params['act_func'],
#                       two_hidden_layers=params['two_hidden_layers'],
#                       num_output_neurons=len(params['data'].training_target_outputs[0]))
    print('Number of hidden neurons: {0}'.format(len(network.layers[0].neurons)))
    print('Number of output neurons: {0}'.format(len(network.layers[-1].neurons)))
    print('Number of total layers: {0}'.format(len(network.layers)))
    
    # test the network
    print("\nBefore training")
    network_module.print_network_error(network, params['data'])
    
    # Train the network
#     print('\nWill train for {0} iterations'.format(params['iterations']))
    trainer = training.Backpropagation()
    trainer.train(network, params['data'].training_inputs, 
                  params['data'].training_target_outputs, 
                  params['learning_rate'], params['momentum_coef'],
                  params['iterations'], params['min_error'])
    
    # test the network
    print("\nAfter training")
#     network_module.print_network_outputs(network, params['data'])
    network_module.print_network_error(network, params['data'])
    print('Trained for {0} iterations'.format(trainer.iterations))
    
    
if __name__ == '__main__':
    # this will be used for timing the code
    start_time = time.clock()
    
    # 1 = XOR
    # 2 = Iris
    # 3 = CT
    # 4 = letter recognition
    project = 2
    
    if project == XOR_PROJECT:
        print("Running XOR test project\n")
    elif project == IRIS_PROJECT:
        print("Running Iris test project\n")
    elif project == CT_PROJECT:
        print("Running CT clinical project\n")
    elif project == LETTER_RECOG_PROJECT:
        print("Running Letter Recognition test project\n")
    
    main(project)

    print("\nCode took %s seconds to run\n" % (str(time.clock() - start_time)))

    
    