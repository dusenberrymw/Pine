'''


@author: dusenberrymw
'''
import csv
import os.path
import random
import re

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
        examples.append([input_vector,target_vector])

    # take 2/3 of items to train for better learning
    training_examples = examples[::3] + examples[1::3]
    testing_examples = examples[2::3]
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
    examples = []
    for row in data:
        # first column contains the target output
        input_vector = [int(x) for x in row[1:]]
        character = row[0].lower() # this is a letter of alphabet
        number = ord(character) - 96 # ascii - 96 = letter number
        target_vector = [0] * 26 # 26 letters
        target_vector[number-1] = 1 # set to 1
        examples.append([input_vector,target_vector])

    # # train on first 16000
    # training_inputs = inputs[:16000]
    # training_target_outputs = target_outputs[:16000]
    # testing_inputs = inputs[16000:]
    # testing_target_outputs = target_outputs[16000:]

    # train on first 500
    training_examples = examples[:16000]
    testing_examples = examples[16000:18000]

    return training_examples, testing_examples


def xor_data():
    """Return a data object containing the training and testing data for the
    XOR logic test project
    """
    data = [ [[0,0],[0]], [[1,0],[1]], [[0,1],[1]], [[1,1],[0]] ]
    training_examples = data[:]
    testing_examples = data[:]
    return training_examples, testing_examples


def and_data():
    """Return a data object containing the training and testing data for the
    XOR logic test project
    """
    data = [ [[0,0],[0]], [[1,0],[0]], [[0,1],[0]], [[1,1],[1]] ]
    training_examples = data[:]
    testing_examples = data[:]
    return training_examples, testing_examples

