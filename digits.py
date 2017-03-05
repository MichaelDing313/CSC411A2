#digits.py
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import pickle
import os
from scipy.io import loadmat

def calculate_output(weights, x):
    ''' This funciton calcuate the output of the nerual nerwork given weights and
        an input. The nerual net is fully connected and each weight corrosponde
        to a connection. For each input x, there are o number of weights, o being
        the number of output.
        
        :param weight: weight matrix of the inputs, dimension lenb(x) by len(o)
        :param x: every input to the nerual net
        
        :returns: nerual net output, though a softmax normalization
        
        TODO:
            none, done

        '''
        
    out = np.dot(x, weights[:784, :]) + weights[-1, :]
    # NOTE: this returns the row of the result, not too sure if we want this
    return softmax(out)
    
# def softmax(y):
#     '''Return the output of the softmax function for the matrix of output y. y
#     is an NxM matrix where N is the number of outputs for a single case, and M
#     is the number of cases'''
#     return exp(y)/tile(sum(exp(y),0), (len(y),1))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
    
def NLL(y, y_):
    return -sum(y_*log(y)) 

def part2():
    ''' This is the function to run part 2 of the assignment
        
        :no params:
        
        :returns: nerual net output, though a softmax normalization
        
        TODO:
            none, done

        '''
        
    np.random.seed(0)
    weights = 2*np.random.random((785, 10))-1
    x = np.random.random((1, 784))
    print(calculateOutput(weights, x))


def cf(x,y,weights):
    # Part 3, neglog prob
    ''' This is the negative log probabability 
        cost function for the nerual network 
        
        :param x: input vector to the network
        :param y: y value to train against, should be 1xn array
        
        :returns: negative log cost fuction output
        
        TODO:
            none, done
        '''
        
    L0 = calculate_output(weights, x)
    return NLL(L0,y) 
    
def dcf(x,y,w):
    # Part3(a) Gradient of the cost funtion    
    ''' This is the gradient function of the entire network, computed step by
        step using the chain rule
        
        :param x: input to the network
        :param y: y value to train against, should be 1xn array
        
        :param w: weight matix that corrospondes to the weights in the nerual net. wij is the weight of ith input to the jth output.
        
        :returns: gradient of the nerual network
        
        TODO:
            verify if this is legit
        '''
    
    # The final equation we seek is:
    # 
    # dC/dWij = SUM(dC/dPj * dPj/dOi * dOi/dhk * dhk/dwij)
    #

    x = calculate_output(w, x)
    dCdO =  x.T - y
    dCdhk =  w[:784, :] * dCdO
    dCdWij = np.dot(w[:784, :].T, dCdhk)
    
    return dCdWij
    
    
    
    
## Part 3b verify gradient using finite differences
M = loadmat("mnist_all.mat")
x = np.random.random((1, 784)) 

h = 1e-5

#pick some random images to look at
# 
# x = collapse_image(act_set[7][3],"resized_all/").T
# x = vstack( (ones((1, x.shape[1])), x))
y = np.matrix([[0.1],[0.1],[0.1],[0.1],[0.1],[0.1],[0.1],[0.9],[0.1],[0.1]]) #linspace(1,1, 10)
theta0 = 2*np.random.random((785, 10))-1
#theta0 = hstack( (ones((1, theta0.shape[0])), theta0))
i = 0
while i < shape(x)[0]:
    if i%100 == 0:
        # test 12 components of the gradient function
        temp = theta0.copy()
        temp[i] -= h
        fake_grad = (cf(x, y, theta0) - cf(x, y, temp)) / h
        real_grad = dcf(x, y, theta0)
        print("Grad:" + str(real_grad[i]) + "   Error:"+ str((fake_grad - real_grad[i])))
    i += 1


    

    
    
if __name__ == __main__:
    # stuff here
    pickle.load(open("snapshot50.pkl", "rb"), encoding="latin1")