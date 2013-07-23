'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import os.path, csv, random, re


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


def parse_data(data_file, only_predict=False):
    """Given a data file, will return a list of training examples
    
    Expects examples file to be of format:
        "[target1[,target2[,...]]] | input1[,input2[,...]]"
    
    """
    p = re.compile('^((?:[\d]+(?:[.][\d]+)?(?:,[\d]+(?:[.][\d]+)?)*)?) [|] ([\d]+(?:[.][\d]+)?(?:,[\d]+(?:[.][\d]+)?)+)$')
    examples = []
    row_index = 1
    for row in data_file:
        row = row.rstrip() # get rid of trailing whitespace
        try:
            match = p.match(row)
            if only_predict: # don't need the targets, even if present
                target_vector = ['']
            else:  # file contains targets
                target_vector = [float(value) for value in match.group(1).split(",")]
            input_vector = [float(value) for value in match.group(2).split(",")]
            examples.append([target_vector,input_vector])
            row_index += 1
        except:
            print("Error on row #{0}: '{1}'".format(row_index, row))
            print("Input must be numerical and in format: [target1[,target2[,...]]] | input1[,input2[,...]]")
            exit()
    random.shuffle(examples)
    return examples
    
    
    
#     data = []
#     for row in data_file:
#         # split in target(s), input(s)
#         row = row.rstrip().split("|")
#         for i in range(len(row)):
#             # split multiple values into a list
#             row[i] = row[i].strip()
#             row[i] = row[i].split(",")
#             for j in range(len(row[i])):
#                 if row[i][j].strip() is '':
#                     continue # no target provided
#                 try:
#                     row[i][j] = int(row[i][j].strip())
#                 except ValueError:
#                     row[i][j] = float(row[i][j].strip())
#         data.append(row)
#     random.shuffle(data)
#     return data


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
    examples = []
    for row in data:
        row_length = len(row)
        # final column contains the target output
        input_vector = ([float(x) for x in row[:row_length-1]])
        output = row[row_length-1]
        if output == 'Iris-setosa':
            target_vector = [1,0,0]
        elif output == 'Iris-versicolor':
            target_vector = [0,1,0]
        elif output == 'Iris-virginica':
            target_vector = [0,0,1]
        examples.append([target_vector, input_vector])
    
    # take 2/3 of items to train for better learning
    training_examples = examples[::3] + examples[1::3]
    testing_examples = examples[2::3]
    
    # Store the data in an object
#     return Data(inputs, target_outputs, training_inputs, 
#                 training_target_outputs, testing_inputs, 
#                 testing_target_outputs)
    return training_examples, testing_examples


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
    examples = []
    for row in data:
        # first column contains the target output
        input_vector = [int(x) for x in row[1:]]
        character = row[0].lower() # this is a letter of alphabet
        number = ord(character) - 96 # ascii - 96 = letter number
        target_vector = [0] * 26 # 26 letters
        target_vector[number-1] = 1 # set to 1
        examples.append([target_vector, input_vector])
    
#     # train on first 16000
#     training_inputs = inputs[:16000]
#     training_target_outputs = target_outputs[:16000]
#     testing_inputs = inputs[16000:]
#     testing_target_outputs = target_outputs[16000:]

    # train on first 500
    training_examples = examples[:16000]
    testing_examples = examples[16000:18000]
    
#     # Store the data in an object
#     return Data(None, None, training_inputs, 
#                 training_target_outputs, testing_inputs, 
#                 testing_target_outputs)
    return training_examples, testing_examples


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
    

def and_data():
    """Return a data object containing the training and testing data for the
    XOR logic test project
    """
    data = [ [[0,0],[0]], [[1,0],[0]], [[0,1],[0]], [[1,1],[1]] ]
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



