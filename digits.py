#digits.py

import pickle
import numpy as np
import matplotlib
    
def calculateOutput(weights, x):
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
    
def softmax(in_array):
    ''' This fuction computes the softmax probability of a input array
        
        :param in_array: The input array to be softmax'd 
        
        :returns: the soft max probabiliy of each of the output vectors
        
        TODO:
            none, done
        '''
    den = sum(np.exp(in_array)
    return in_array / den

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


def cf(x,y):
    # Part 3, neglog prob
    ''' This is the negative log probabability 
        cost function for the nerual network 
        
        :param w: weight vector for each element 1xn , use all 1 if no weight
        :param y: y value to train against, should be 1xn array
        
        :returns: negative log cost fuction output
        
        TODO:
            none, done
        '''
    a = -sum(y * np.log(softmax(x)))
    


    

    
    
if __name__ = __main__:
    # stuff here
    pickle.load(open("snapshot50.pkl", "rb"), encoding="latin1")