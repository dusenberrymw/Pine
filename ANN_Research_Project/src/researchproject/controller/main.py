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
import cProfile
import copy
from multiprocessing import Process, Queue, cpu_count

# constants for the different projects
XOR_PROJECT = 1
IRIS_PROJECT = 2
CT_PROJECT = 3

def main(project):
    """Starting point for this program
    
    Will build a default list of parameters, get the data for the specified
    project, then build and train networks in parallel, trying different
    configurations
    
    """
    # Build the params for the project
#     params = build_project_params(project)
#     test_run(params)
    parallel_run()


def parallel_run():
    # Build the params for the project
    params = build_project_params(project)
    
    # Determine the number of processes
    num_processes = cpu_count()
    
    # Create the master network
    master_network = Network(len(params['data'].training_inputs[0]), 
                      params['act_func'],
                      params['num_hidden_layer1_neurons'], 
                      params['two_hidden_layers'],
                      params['num_hidden_layer2_neurons'],
                      len(params['data'].training_target_outputs[0]))
    trainer = training.Backpropagation()
    
    # Create the return queue to store the training networks that are returned
    #    from each process
    results_queue = Queue() # this is the multiprocessing queue

    # Train the training networks on a subset of the data
    #    this is where the parallelization will occur
    #    -Note: this process counts as one of the processes, so be sure to
    #        have this one do work as well
    jobs = [Process(target=trainer.parallel_train, 
                    args=(master_network, i, params, num_processes, results_queue))
                    for i in range(num_processes-1)]
    # start the other processes
    for job in jobs: job.start()
    # while those processes are running, perform this process's work as well
    trainer.parallel_train(copy.deepcopy(master_network), num_processes-1, params, 
                   num_processes, results_queue)
    # Now wait for the other processes
    for job in jobs: job.join()
    
    # retrieve the trained networks
    training_networks = [results_queue.get() for _ in range(num_processes)]
    
    # Now average out the training networks into the master network
    for i, layer in enumerate(master_network.layers):
        for j, neuron in enumerate(layer.neurons):
            for k in range(len(neuron.weights)):
                weight_sum = 0
                for network in training_networks:
                    weight_sum += network.layers[i].neurons[j].weights[k]
                neuron.weights[k] = weight_sum/num_processes

    # check the error on the master network
    print("\nMaster Network:")
    network_module.print_network_error(master_network, params['data'])


def build_project_params(project):
    """Create the default parameters for this project"""
    # create a dictionary to store all params
    params = {}
    
    # Set up default parameters for this project
    params['act_func'] = training.SigmoidActivationFunction()
    params['num_hidden_layer1_neurons'] = None
    params['num_hidden_layer2_neurons'] = None
    params['two_hidden_layers'] = False
    params['learning_rate'] = 0.1
    params['momentum_coef'] = 0.5
    params['min_error'] = 0.0001
    params['iterations'] = 10000
    
    # Get the project's data and any overrides to the defaults
    if project == XOR_PROJECT:
        params['data'] = data_module.xor_data()
        params['learning_rate'] = 0.7
        params['momentum_coef'] = 0.1
    elif project == IRIS_PROJECT:
        params['data'] = data_module.iris_data()
        params['learning_rate'] = 0.2
        params['momentum_coef'] = 0.5
    elif project == CT_PROJECT:
        params['data'] = data_module.ct_data()
    
    return params


def test_run(params):
    """This creates a network, tests it, trains it, and retests it"""
    # Create the network
    network = Network(len(params['data'].training_inputs[0]), params['act_func'],
                      two_hidden_layers=params['two_hidden_layers'],
                      num_output_neurons=len(params['data'].training_target_outputs[0]))
    print('Number of hidden neurons: {0}'.format(len(network.layers[0].neurons)))
    print('Number of output neurons: {0}'.format(len(network.layers[-1].neurons)))
    print('Number of total layers: {0}'.format(len(network.layers)))
    
    # test the network
    print("\nBefore training")
    network_module.print_network_error(network, params['data'])
    
    # Train the network
    print('\nWill train for {0} iterations'.format(params['iterations']))
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
    project = 2
    
    if project == XOR_PROJECT:
        print("Running XOR test project")
    elif project == IRIS_PROJECT:
        print("Running Iris test project")
    elif project == CT_PROJECT:
        print("Running CT clinical project")
    
    print()
    main(project)

    print("\nCode took %s seconds to run\n" % (str(time.clock() - start_time)))

    
    