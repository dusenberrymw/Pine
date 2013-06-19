'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import os.path
import csv


class Data(object):
    """This is the class that will serve as the model of supplied data
    as well as the network and any other useful information
    """
    
    def __init__(self, training_inputs=None, training_target_outputs=None, 
                 testing_inputs=None, testing_target_outputs=None):
        self.training_inputs = training_inputs
        self.training_target_outputs = training_target_outputs
        self.testing_inputs = testing_inputs
        self.testing_target_outputs = testing_target_outputs
        self.network = None


def ct_data():
    """Return a data object containing the training and testing data for the
    CT project
    """
    # Set up the inputs and target outputs
    inputs = []
    target_outputs = []
    with open(os.path.join(os.path.dirname(__file__), 
                           './data/CT_Data_Edited.csv'), 
                           newline='') as data_file:
            data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
            for row in data_reader:
                row_length = len(row)
                temp_list = [int(x) for x in row]
                inputs.append(temp_list[:row_length-1])
                target_outputs.append(temp_list[row_length-1:row_length])
    
            data_file.close()

    # build the training and testing sets
    training_inputs = inputs[::2] # take every other item
    training_target_outputs = target_outputs[::2]
    testing_inputs = inputs[1::2] # take the other items
    testing_target_outputs = target_outputs[1::2]
    
    # Store the data in an object
    return Data(training_inputs, training_target_outputs,
                testing_inputs, testing_target_outputs)
    


def iris_data():
    """Return a data object containing the training and testing data for the
    iris test project
    """
    # Get the inputs and target outputs
    inputs = []
    target_outputs = []
    with open(os.path.join(os.path.dirname(__file__), 
                           './data/iris.data.txt'), 
                           newline='') as data_file:
            data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
            for row in data_reader:
                row_length = len(row)
                inputs.append([float(x) for x in row[:row_length-1]])
                output = row[row_length-1]
                if output == 'Iris-setosa':
                    output = [1,0,0]
                elif output == 'Iris-versicolor':
                    output = [0,1,0]
                elif output == 'Iris-virginica':
                    output = [0,0,1]
                target_outputs.append(output)
    
            data_file.close()

    # build the training and testing sets
    training_inputs = inputs[::2] # take every other item
    training_target_outputs = target_outputs[::2]
    testing_inputs = inputs[1::2] # take the other items
    testing_target_outputs = target_outputs[1::2]
    
    # Store the data in an object
    return Data(training_inputs, training_target_outputs,
                testing_inputs, testing_target_outputs)


def xor_data():
    """Return a data object containing the training and testing data for the
    XOR logic test project
    """
    training_inputs = [[0,0],[1,0],[0,1],[1,1]]
    training_target_outputs = [[0],[1],[1],[0]]
    testing_inputs = training_inputs
    testing_target_outputs = training_target_outputs
    
    # Store the data in an object
    return Data(training_inputs, training_target_outputs,
                testing_inputs, testing_target_outputs)



