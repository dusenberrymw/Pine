#! /usr/bin/env python3
'''
Created on April 28th, 2014

@author: dusenberrymw
'''
import os
import subprocess

pineCLI_path = os.path.realpath("./bin/pineCLI.py")
install_path = "/usr/local/bin/pine2"

cmd = ["ln", "-s", pineCLI_path, install_path]

print("Trying: \'" + ' '.join(cmd) + "\'")

try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError as error:
    print("An error occurred")
    exit()

print("Success: Pine installed to /usr/local/bin/pine as a soft link to ./bin/pineCLI.py")
print("To run Pine from anywhere, type \'pine\' at the command line")
