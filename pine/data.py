'''
Created on Jun 6, 2013

@author: dusenberrymw
'''
import re

def parse_data(data_file, only_predict=False):
    """Given a data file, will return a list of training examples

    Expects examples file to be of format:
        "[target1[,target2[,...]]] | input1[,input2[,...]]"

    """
    # Make a regex that looks for the above pattern, and captures the
    #    targets list (separated with commas) and inputs list (separated
    #    with commas).
    # Note: The '?:' part causes the group (which is defined within the
    #    parentheses) to NOT be backreferenced, or in other words, a match
    #    to the regex will not save this group.  We use this so that we can
    #    apply '*' or '?' to a certain group, but in the end we only want the
    #    entire targets group and entire inputs group.
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
            examples.append([input_vector,target_vector])
            row_index += 1
        except:
            print("Error on row #{0}: '{1}'".format(row_index, row))
            print("Input must be numerical and in format: [target1[,target2[,...]]] | input1[,input2[,...]]")
            exit()
    return examples
