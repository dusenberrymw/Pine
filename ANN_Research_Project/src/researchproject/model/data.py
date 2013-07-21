'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import os.path, csv, random


class Data(object):
    """This is the class that will serve as the model of supplied data
    as well as the network and any other useful information
    """
    
    def __init__(self, inputs, target_outputs, training_inputs,
                 training_target_outputs, testing_inputs, 
                 testing_target_outputs):
        self.inputs = inputs
        self.target_outputs = target_outputs
        self.training_inputs = training_inputs
        self.training_target_outputs = training_target_outputs
        self.testing_inputs = testing_inputs
        self.testing_target_outputs = testing_target_outputs


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
            data = [row for row in data_reader]
            data_file.close()
    random.shuffle(data)
    for row in data:
        row_length = len(row)
        temp_list = [int(x) for x in row]
        inputs.append(temp_list[:row_length-1])
        target_outputs.append(temp_list[row_length-1:row_length])

    # build the training and testing sets
    training_inputs = inputs[::2] # take every other item
    training_target_outputs = target_outputs[::2]
    testing_inputs = inputs[1::2] # take the other items
    testing_target_outputs = target_outputs[1::2]
    
    # Store the data in an object
    return Data(inputs, target_outputs, training_inputs, 
                training_target_outputs, testing_inputs, 
                testing_target_outputs)


def iris_data():
    """Return a data object containing the training and testing data for the
    iris test project
    """
    # Fetch the data from the file
    with open(os.path.join(os.path.dirname(__file__), 
                           './data/iris.data.txt'), 
                           newline='') as data_file:
            data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
            data = [row for row in data_reader]
            data_file.close()
    
    # Pull the inputs and target outputs out of the data
    random.shuffle(data)
    inputs = []
    target_outputs = []
    for row in data:
        row_length = len(row)
        # final column contains the target output
        inputs.append([float(x) for x in row[:row_length-1]])
        output = row[row_length-1]
        if output == 'Iris-setosa':
            output = [1,0,0]
#             output = [-1]
        elif output == 'Iris-versicolor':
            output = [0,1,0]
#             output = [0]
        elif output == 'Iris-virginica':
            output = [0,0,1]
#             output = [1]
        target_outputs.append(output)
    
    # build the training and testing sets
#     training_inputs = inputs[::2] # take every other item
#     training_target_outputs = target_outputs[::2]
#     testing_inputs = inputs[1::2] # take the other items
#     testing_target_outputs = target_outputs[1::2]
    
    # take 2/3 of items to train for better learning
    training_inputs = inputs[::3] + inputs[1::3]
    training_target_outputs = target_outputs[::3] + target_outputs[1::3]
    testing_inputs = inputs[2::3] # take the other 1/3 of items
    testing_target_outputs = target_outputs[2::3]
    
    # Store the data in an object
    return Data(inputs, target_outputs, training_inputs, 
                training_target_outputs, testing_inputs, 
                testing_target_outputs)


def letter_recognition_data():
    """Return a data object containing the training and testing data for the
    letter recognition test project
    """
    # Fetch the data from the file
    with open(os.path.join(os.path.dirname(__file__), 
                           './data/letter_recognition.data.txt'), 
                           newline='') as data_file:
            data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
            data = [row for row in data_reader]
            data_file.close()
    
    # Pull the inputs and target outputs out of the data
    random.shuffle(data)
    inputs = []
    target_outputs = []
    for row in data:
        # first column contains the target output
        inputs.append([int(x) for x in row[1:]])
        character = row[0].lower() # this is a letter of alphabet
        number = ord(character) - 96 # ascii - 96 = letter number
        output = [0] * 26 # 26 letters
        output[number-1] = 1 # set to 1
        target_outputs.append(output)
#         target_outputs.append([number/26])
    
#     # train on first 16000
#     training_inputs = inputs[:16000]
#     training_target_outputs = target_outputs[:16000]
#     testing_inputs = inputs[16000:]
#     testing_target_outputs = target_outputs[16000:]

    # train on first 16000
    training_inputs = inputs[:500]
    training_target_outputs = target_outputs[:500]
    testing_inputs = inputs[500:700]
    testing_target_outputs = target_outputs[500:700]
    
    # Store the data in an object
    return Data(None, None, training_inputs, 
                training_target_outputs, testing_inputs, 
                testing_target_outputs)


def xor_data():
    """Return a data object containing the training and testing data for the
    XOR logic test project
    """
    data = [ [[0,0],[0]], [[1,0],[1]], [[0,1],[1]], [[1,1],[0]] ]
    random.shuffle(data)
    inputs = []
    target_outputs = []
    for row in data:
        inputs.append(row[0])
        target_outputs.append(row[1])
    training_inputs = inputs
    training_target_outputs = target_outputs
    testing_inputs = training_inputs
    testing_target_outputs = training_target_outputs
    
    # Store the data in an object
    return Data(inputs, target_outputs, training_inputs, 
                training_target_outputs, testing_inputs, 
                testing_target_outputs)



