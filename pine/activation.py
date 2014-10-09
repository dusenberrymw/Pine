'''
Created on Oct 9, 2014

@author: dusenberrymw
'''
import math

def isValidFunction(name):
    functionNames = ["logistic", "tanh", "linear"]
    return name.lower() in functionNames


class Logistic(object):
    """The Logistic (Sigmoid) activation function, which is one of the
        possibilities that can be used by the network

    """
    def __init__(self):
        self.name = "logistic"

    def activate(self, input_value):
        """Run the input value through the sigmoid function
        This will only return values between 0 and 1
        """
        try:
            return 1.0 / (1 + math.exp(-1.0*input_value))
        except OverflowError:
            # bound the numbers if there is an overflow
            if input_value < 0:
                return 0.0000000000001 #logistic func goes to 0 for small x
            else:
                return 0.9999999999999

    def derivative(self, fx):
        """Calculate derivative of logistic function, given an output F(x)

        Given F is the logistic (sigmoid) function, the derivative
            F'(x) = F(x) * (1 - F(x))

        F(x) will be passed in for efficiency
        """
        return fx*(1.0-fx)

    def inverse(self, input_value):
        """This will produce the inverse of the sigmoid function, which is
        useful in determining the original value before activation
        """
        return math.log(input_value/(1-input_value))

    def cost(self, hypothesis_output, target_output):
        """Cost function, J, of a node using the logistic activation function

        J_theta(h_theta(x), y) = -log(h_theta(x))   if y = 1
                                 -log(1-h_theta(x)) if y = 0
            where h_theta(x) is the hypothesis (computed output) of the
                node evaluated with respect to theta (parameter/weight vector)
                evaluated at input x,
            and J_theta is the "error" (cost) of the node with respect to
                theta (parameter/weight vector) evaluated at the hypothesis 
                of x given the target value y

        This cost function essentially allows for no error if the hypothesis
            is equal to the target y, and high error otherwise

        """
        y = target_output
        h_x = hypothesis_output
        J = -y*math.log10(h_x)-(1-y)*math.log10(1-h_x)
        return J

    def cost_derivative(self, hypothesis_output, target_output):
        """Partial derivative of the cost function, J, of a neuron using the
        logistic activation function

        This will basically determine how much the hypothesis (output of the
            neuron) contributed to the cost ("error") of the neuron

        Note: math.log is ln

        """
        y = target_output
        h_x = hypothesis_output
        dJ = ((y-1)/((h_x-1)*math.log(10))) - (y/(h_x*math.log(10))) 
        return dJ


class Tanh(object):
    """The Tanh (Logistic spinoff) activation function, which is one of the
        possibilities that can be used by the network

    """
    def __init__(self):
        self.name = "tanh"

    def activate(self, input_value):
        """Run the input value through the tanh function"""
        #return math.tanh(input_value)
        return ((math.exp(input_value)-math.exp(-input_value)) /
               (math.exp(input_value)+math.exp(-input_value)))

    def derivative(self, fx):
        """Some training will require the derivative of the tanh function"""
        return (1.0-fx) * (1.0+fx)

    def inverse(self, input_value):
        """This will produce the inverse of the tanh function, which is
        useful in determining the original value before activation
        """
        return math.atanh(input_value)

    def cost(self, hypothesis_output, target_output):
        """Cost function, J, of a node using the tanh activation function

        J_theta(h_theta(x), y) = (1/2)*(|h_theta(x)-y|^2)

        """
        y = target_output
        h_x = hypothesis_output
        # return (1/2)*(math.fabs(h_x-y)**2)
        return (1/2)*((h_x-y)**2)

    def cost_derivative(self, hypothesis_output, target_output):
        """Partial derivative of the cost function, J, of a neuron using the
        tanh activation function

        This will basically determine how much the hypothesis (output of the
            neuron) contributed to the cost ("error") of the neuron

        """
        y = target_output
        h_x = hypothesis_output
        dJ = h_x - y
        return dJ


class Linear(object):
    """The Linear activation function, which is one of the
        possibilities that can be used by the network

    """
    def __init__(self):
        self.name = "linear"

    def activate(self, input_value):
        """In the linear function, f(z) = z"""
        return input_value

    def derivative(self, fx):
        """Some training will require the derivative of the linear function
        which is just 1
        """
        return 1

    def inverse(self, input_value):
        """This will produce the inverse of the linear function, which is
        useful in determining the original value before activation
        """
        return input_value

    def cost(self, hypothesis_output, target_output):
        """Cost function of a node using the Linear activation function

        cost_theta(h_theta(x), y) = (1/2)*((h_theta(x)-y)^2)

        """
        y = target_output
        h_x = hypothesis_output
        return (1/2)*((h_x-y)**2)

    def cost_derivative(self, hypothesis_output, target_output):
        """Partial derivative of the cost function, J, of a neuron using the
        linear activation function

        This will basically determine how much the hypothesis (output of the
            neuron) contributed to the cost ("error") of the neuron

        """
        y = target_output
        h_x = hypothesis_output
        dJ = h_x - y
        return dJ
