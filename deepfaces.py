################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import tensorflow as tf

from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]



################################################################################
#Read Image, and change to BGR


im1 = (imread("laska.png")[:,:,:3]).astype(float32)
im1 = im1 - mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

im2 = (imread("poodle.png")[:,:,:3]).astype(float32)
im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]


################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def log_event(log_string, level):
    '''
    This function handles info display in console, mainly here to unify logging
    and let me control the level of logging displayed
    
    :param log_string:  Log message to be displayed
    :param level:       Severity of event, higher the number more important
    '''
    # Assign preset log level to number
    if log_level == "ALL": log_lvl_local = 0
    if log_level == "SHORT": log_lvl_local = 1
    if log_level == "INFO": log_lvl_local = 2
    if log_level == "WARNING": log_lvl_local = 3
    if log_level == "NONE": log_lvl_local = 999
    
    # Check log severity and skip function of no logging needed
    if level < log_lvl_local: return
    
    # Case if we want short logging, plrint a dot for each low level event
    # instead of printing full message.
    # Log level 1 allow full message when "ALL" and dot output when "SHORT"
    if (log_lvl_local == 1 and level == 1):
        print '.' ,
    else:
        print(log_string)
        
# Make folders for files to be stored
def make_folders(folder_list):
    log_event(" Generating Folders ",2)
    for directory in folder_list:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)          
                log_event("  PASS  " + directory,1)  
            except:
                log_event("  FAIL  " + directory,3)
        else:
            log_event("  ALREADY EXIST  " + directory,2)

    log_event(" Folder Creation Complete ",2)

## Fetch images from intertubes
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result
                
def fetch_image_URL_mod(actor_list, output_folder, source_list, output_list):
    testfile = urllib.URLopener()            
    out_file = open(output_list,'w')
    
    log_event(" Fetching Images From Web ",2)
    for a in actor_list:
        name = a.split()[1].lower()
        i = 0
        for line in open(source_list):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                timeout(testfile.retrieve, (line.split()[4], output_folder+filename), {}, 15)
                
                # If there is no file, i.e theimage was not obtained
                if not os.path.isfile(output_folder+filename):
                    log_event(" Skip - Timed Out ",3)
                    continue
                    
                # Check if file is a valid image, eliminate semi dead links (404s)
                if (imghdr.what(output_folder+filename) == None):
                    log_event(" Skip - Image Invalid ",3)
                    continue
                    
                #import pdb; pdb.set_trace()
                # Check if the downloaded image matches the checksum
                f = open(output_folder+filename, "rb")
                buf = f.read()
                f.close()
                if (hashlib.sha256(buf).hexdigest() != line.split()[6]):
                    log_event(" Skip - Bad Hash ",3)
                    try:
                        os.remove(output_folder+filename)
                    except:
                        log_event(" Bad hash, delete FAIL ",3)
                    continue
    
                # Only run if file is valid
                
                # Output file with crop info on the side
                out_file.write(a + ' ' + filename + ' ' + line.split()[5] + ' ' + line.split()[4] +'\n')
                log_event("  PASS  " + filename,1)
                i += 1
                log_event(" Retrieved: " + filename,1)

    out_file.close()
    log_event(" Image Fetch Complete ",2)
    
# Crop Images
def crop_images(act, input_folder, output_folder, ref_file, output_list):
    '''
    This Function takes in a folder of uncropped images, crops them and output
    the result into another (or same) folder
    act             - list of actors to convert
    input_folder    - folder location for file input
    output_folder   - folder location for output file
    ref_file        - location of reference file that holds info
    '''
    log_event(" Cropping Images ",2)
    out_file = open(output_list,'w')
    for a in act:
        for line in open(ref_file):
            if a in line:
                filename = line.split()[2] # Extract file name to crop
                try:
                    cc = line.split()[3].split(",") # Extract crop coords
                    im = imread(input_folder + filename)
                    im_crop = im[int(cc[1]):int(cc[3]), int(cc[0]):int(cc[2])]                    
                    scipy.misc.imsave(output_folder + filename,im_crop)
                except:                    
                    log_event("  FAIL  Cropping Error" + filename,3)
                    continue
                    
                # If there is no file, i.e copy failed
                if not os.path.isfile(output_folder + filename):
                    log_event("  FAIL  No Output" + filename,3)
                    continue
                
                log_event("  PASS  " + filename,1)
                out_file.write(line)

    log_event(" Cropping Complete ",2)

def resize_227x227(act, input_folder, output_folder, ref_file, output_list):
    '''
    This Function takes in a folder of images, and convert
    them into 227x227 in size
    act             - list of actors to convert
    input_folder    - folder location for file input
    output_folder   - folder location for output file
    ref_file        - location of reference file that holds info
    '''
    log_event(" Resizing Images ",2)
    out_file = open(output_list,'w')
    for a in act:
        for line in open(ref_file):
            if a in line:
                filename = line.split()[2] # Extract file name to convert
                try:
                    im = imread(input_folder + filename)
                    
                    im_out = imresize(im, (227,227))
    
                    scipy.misc.imsave(output_folder + filename,im_out)
                except:                    
                    log_event("  FAIL  " + filename,3)
                    continue
                    
                # If there is no file, i.e copy failed
                if not os.path.isfile(output_folder + filename):
                    log_event("  FAIL  " + filename,3)
                    continue
                
                log_event("  PASS  " + filename,1)
                out_file.write(line)
    log_event(" Resizing Complete ",2)
    
def pick_sets(act, ref_file):
    '''
    This function picks the training, validation and test set for a given actor
    as per requirment, 100 images for training, 10 each for validation and test
    Create file name set for each actor, in the format below:
    [   [[actor_name],[Traning_set],[validation_set],[test_set]], 
        [[actor_name],[Traning_set],[validation_set],[test_set]], 
        [...] ]
    
    act         - actor name list
    ref_file    - reference file to all image filenames
    '''
    # Seed random generator
    np.random.seed(411411411)
    
    log_event(" Picking Data Sets ",2)
    i = 0
    output_set = []
    
    for a in act:
        output_set.append([[],[],[],[]])
        output_set[i][0] = a
        temp_list = []
        for line in open(ref_file):
            if a in line:
                temp_list.append(line.split()[2]) # Extract file name for actor a
        try:
            np.random.shuffle(temp_list)
            output_set[i][1] = temp_list[:60]
            output_set[i][2] = temp_list[60:70]
            output_set[i][3] = temp_list[70:100]
        except:
            log_event("  FAIL  " + a,3)
            
        log_event("  PASS  " + a,2)
        
        i+=1
    
    log_event(" Data Set Selection Complete ",2)
    return output_set

def alexify(in_set, source_folder):
    '''
    This function reads in the images and convert them into alex net compatible
    input variables
    
    in_set          - array of input image names to convert to data set
    source_folder   - the folder of which files are stored

    '''
    log_event(" Alexifying Images ",2)
    i = 0
    x = empty((size(in_set),227,227,3))
    
    for j in in_set:
        
        im1 = (imread(source_folder + j)[:,:,:3]).astype(float32)
        im1 = im1 - mean(im1)
        im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
        
        x[i] = im1
        log_event("  PASS  " + j,1)
        i+=1
    
    log_event(" Alexifying Images ",2)
    return x
    
def get_train(id):
    '''
    This function take in the input data set and return two arrays for training input and output
    
    :param id:  input data set, in the format:    
                [   [[actor_name],[Traning_set],[validation_set],[test_set]], 
                [[actor_name],[Traning_set],[validation_set],[test_set]], 
                [...] ]
    
    :returns:   two arrays, array of images and corrosponding labels
    
    '''
    ts = 60
    
    train_y = np.append(linspace(1,1, ts), linspace(0,0, 5*ts))                                      # bill = [1,0,0,0,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, ts), linspace(1,1, 1*ts)), linspace(0,0, 4*ts)), train_y))   # steve = [0,1,0,0,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, 2*ts), linspace(1,1, 1*ts)), linspace(0,0, 3*ts)), train_y))    # alec = [0,0,1,0,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, 3*ts), linspace(1,1, 1*ts)), linspace(0,0, 2*ts)), train_y))    # kristin = [0,0,0,1,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, 4*ts), linspace(1,1, 1*ts)), linspace(0,0, 1*ts)), train_y))    # america = [0,0,0,0,1,0]
    train_y = vstack( (np.append(linspace(0,0, 5*ts), linspace(1,1, 1*ts)), train_y))    # fran = [0,0,0,0,0,1]
    
    train_x = alexify(act_set[4][1],"resized_all/")          # Bill Hader
    train_x = np.append(train_x, alexify(act_set[5][1],"resized_all/"), axis=0)    # Steve Carell
    train_x = np.append(train_x, alexify(act_set[3][1],"resized_all/"), axis=0)    # Alec Baldwin
    train_x = np.append(train_x, alexify(act_set[2][1],"resized_all/"), axis=0)    # Kristin Chenoweth
    train_x = np.append(train_x, alexify(act_set[1][1],"resized_all/"), axis=0)    # America Ferrera
    train_x = np.append(train_x, alexify(act_set[0][1],"resized_all/"), axis=0)    # Fran Drescher
    
    return train_x, train_y.T

def get_vali(id):
    '''
    This function take in the input data set and return two arrays for testing
    
    :param id:  input data set, in the format:    
                [   [[actor_name],[Traning_set],[validation_set],[test_set]], 
                [[actor_name],[Traning_set],[validation_set],[test_set]], 
                [...] ]
    
    :returns:   two arrays, array of images and corrosponding labels
    
    '''
    ts = 10
    
    train_y = np.append(linspace(1,1, ts), linspace(0,0, 5*ts))                                      # bill = [1,0,0,0,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, ts), linspace(1,1, 1*ts)), linspace(0,0, 4*ts)), train_y))   # steve = [0,1,0,0,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, 2*ts), linspace(1,1, 1*ts)), linspace(0,0, 3*ts)), train_y))    # alec = [0,0,1,0,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, 3*ts), linspace(1,1, 1*ts)), linspace(0,0, 2*ts)), train_y))    # kristin = [0,0,0,1,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, 4*ts), linspace(1,1, 1*ts)), linspace(0,0, 1*ts)), train_y))    # america = [0,0,0,0,1,0]
    train_y = vstack( (np.append(linspace(0,0, 5*ts), linspace(1,1, 1*ts)), train_y))    # fran = [0,0,0,0,0,1]
    
    train_x = alexify(act_set[4][2],"resized_all/")          # Bill Hader
    train_x = np.append(train_x, alexify(act_set[5][2],"resized_all/"), axis=0)    # Steve Carell
    train_x = np.append(train_x, alexify(act_set[3][2],"resized_all/"), axis=0)    # Alec Baldwin
    train_x = np.append(train_x, alexify(act_set[2][2],"resized_all/"), axis=0)    # Kristin Chenoweth
    train_x = np.append(train_x, alexify(act_set[1][2],"resized_all/"), axis=0)    # America Ferrera
    train_x = np.append(train_x, alexify(act_set[0][2],"resized_all/"), axis=0)    # Fran Drescher
    
    return train_x, train_y.T
    
def get_test(id):
    '''
    This function take in the input data set and return two arrays for testing
    
    :param id:  input data set, in the format:    
                [   [[actor_name],[Traning_set],[validation_set],[test_set]], 
                [[actor_name],[Traning_set],[validation_set],[test_set]], 
                [...] ]
    
    :returns:   two arrays, array of images and corrosponding labels
    
    '''
    ts = 30
    
    train_y = np.append(linspace(1,1, ts), linspace(0,0, 5*ts))                                      # bill = [1,0,0,0,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, ts), linspace(1,1, 1*ts)), linspace(0,0, 4*ts)), train_y))   # steve = [0,1,0,0,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, 2*ts), linspace(1,1, 1*ts)), linspace(0,0, 3*ts)), train_y))    # alec = [0,0,1,0,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, 3*ts), linspace(1,1, 1*ts)), linspace(0,0, 2*ts)), train_y))    # kristin = [0,0,0,1,0,0]
    train_y = vstack( (np.append(np.append(linspace(0,0, 4*ts), linspace(1,1, 1*ts)), linspace(0,0, 1*ts)), train_y))    # america = [0,0,0,0,1,0]
    train_y = vstack( (np.append(linspace(0,0, 5*ts), linspace(1,1, 1*ts)), train_y))    # fran = [0,0,0,0,0,1]
    
    train_x = alexify(act_set[4][3],"resized_all/")          # Bill Hader
    train_x = np.append(train_x, alexify(act_set[5][3],"resized_all/"), axis=0)    # Steve Carell
    train_x = np.append(train_x, alexify(act_set[3][3],"resized_all/"), axis=0)    # Alec Baldwin
    train_x = np.append(train_x, alexify(act_set[2][3],"resized_all/"), axis=0)    # Kristin Chenoweth
    train_x = np.append(train_x, alexify(act_set[1][3],"resized_all/"), axis=0)    # America Ferrera
    train_x = np.append(train_x, alexify(act_set[0][3],"resized_all/"), axis=0)    # Fran Drescher
    
    return train_x, train_y.T


### ALEX NET IS BELOW
xdim = (227,227,3)

x = tf.placeholder(tf.float32, (None,) + xdim)


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


# #conv5
# #conv(3, 3, 256, 1, 1, group=2, name='conv5')
# k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
# conv5W = tf.Variable(net_data["conv5"][0])
# conv5b = tf.Variable(net_data["conv5"][1])
# conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
# conv5 = tf.nn.relu(conv5_in)
# 
# #maxpool5
# #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
# k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
# maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
# 
# #fc6
# #fc(4096, name='fc6')
# fc6W = tf.Variable(net_data["fc6"][0])
# fc6b = tf.Variable(net_data["fc6"][1])
# fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
# 
# #fc7
# #fc(4096, name='fc7')
# fc7W = tf.Variable(net_data["fc7"][0])
# fc7b = tf.Variable(net_data["fc7"][1])
# fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
# 
# #fc8
# #fc(1000, relu=False, name='fc8')
# fc8W = tf.Variable(net_data["fc8"][0])
# fc8b = tf.Variable(net_data["fc8"][1])
# fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
# 
# 
# #prob
# #softmax(name='prob'))
# prob = tf.nn.softmax(fc8)
# 
# 
# t = time.time()
# output = sess.run(prob, feed_dict = {x:[im1,im2]})


# conv4_size = int(prod(conv4.get_shape()[1:]))
# ex1W = tf.Variable(tf.random_normal([conv4_size, 6], stddev=0.0001, seed=411))
# ex1b = tf.Variable(tf.random_normal([6], stddev=0.0001, seed=411))
# wx1 = tf.matmul(tf.reshape(conv4, [-1, conv4_size]), ex1W)+ex1b
#     
# y = tf.nn.softmax(layer2)
# y_ = tf.placeholder(tf.float32, [None, 6])






################################################################################

#Output:

# 
# for input_im_ind in range(output.shape[0]):
#     inds = argsort(output)[input_im_ind,:]
#     print("Image", input_im_ind)
#     for i in range(5):
#         print(class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]])
# 
# print(time.time()-t)


## INIT CODE
__name__ = "__main__"
# Set log level, for convience, avaliable options are:
# ALL, SHORT, INFO, WARNING, NONE
log_level = "SHORT"

if __name__ == "__init__":
    # Execution start here
    
    ## Part 1 Gather Data
    # Download, filter, crop and resize images to be used
    
    
    #uncomment all these code below to run entire process
    files = ["facescrub_actors.txt","facescrub_actresses.txt"]
    newfile = open('combined_face_scrub.txt', 'w')
    newfile.write( ''.join([open(f).read() for f in files]))
    newfile.close
    
    act = list(set([a.split("\t")[0] for a in open("combined_face_scrub.txt").readlines()]))
    make_folders(('uncropped_all/','uncropped_all','cropped_all','grey_all','resized_all'))
    fetch_image_URL_mod(act, "uncropped_all/","combined_face_scrub.txt","uncropped_all/0_subset_crop_info.txt")
    crop_images(act, "uncropped_all/","cropped_all/","uncropped_all/0_subset_crop_info.txt","cropped_all/0_output_info.txt")
    convert_grey(act, "cropped_all/","grey_all/","uncropped_all/0_subset_crop_info.txt")
    resize_32x32(act, "grey_all/","resized_all/","cropped_all/0_output_info.txt","resized_all/0_set_info.txt")
    
    # end of init code
    
if __name__ == "__main__":
    #t = int(time.time())
    t = 1454219613
    print "t=", t
    random.seed(t)
    
    # Pick the data set to be used for each actor
    
    #Build actor list
    act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    
    act_set = pick_sets(act, "resized_all/0_set_info.txt")
    print('actor order are:')
    actor_order = []
    count = 0
    for i in act_set:
        print(str(count) + ':' + i[0])
        actor_order.append(i[0])
        count += 1
    print(actor_order)

    
    import tensorflow as tf
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    train_x = get_train(act_set)
    output = sess.run(conv4, feed_dict = {x:[train_x]})
