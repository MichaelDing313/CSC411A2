## Initial Imports and things

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from numpy import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import scipy
import matplotlib.image as mpimg
from matplotlib.pyplot import *
import os
from scipy.ndimage import filters
import urllib
import imghdr
import shutil
import hashlib


import cPickle

import os
from scipy.io import loadmat

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

def convert_grey(act, input_folder, output_folder, ref_file):
    '''
    This Function takes in a folder rgb images, and convert
    them into greyscale images
    act             - list of actors to convert
    input_folder    - folder location for file input
    output_folder   - folder location for output file
    ref_file        - location of reference file that holds info
    '''
    log_event(" Gryscalfying Images ",2)
    for a in act:
        for line in open(ref_file):
            if a in line:
                filename = line.split()[2] # Extract file name to convert
                try:
                    im = imread(input_folder + filename)
                    
                    im_out = np.dot(im[...,:3], [0.2989, 0.587, 0.114])
                    
                    #r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]
                    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  
                    #im_out = gray/255

                    scipy.misc.imsave(output_folder + filename, im_out)
                except:                    
                    log_event("  FAIL  " + filename,3)
                    continue
                    
                # If there is no file, i.e copy failed
                if not os.path.isfile(output_folder + filename):
                    log_event("  FAIL  " + filename,3)
                    continue
                
                log_event("  PASS  " + filename,1)
    log_event(" Gryscalfying Complete ",2)
    
def resize_32x32(act, input_folder, output_folder, ref_file, output_list):
    '''
    This Function takes in a folder of images, and convert
    them into 32x32 in size
    act             - list of actors to convert
    input_folder    - folder location for file input
    output_folder   - folder location for output file
    ref_file        - location of reference file that holds info
    '''
    log_event(" Shrinking Images ",2)
    out_file = open(output_list,'w')
    for a in act:
        for line in open(ref_file):
            if a in line:
                filename = line.split()[2] # Extract file name to convert
                try:
                    im = imread(input_folder + filename)
                    
                    im_out = imresize(im, (32,32))
    
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
    log_event(" Shrinking Complete ",2)
    
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

def collapse_image(in_set, source_folder):
    '''
    This function collapses 32x32 image file into a single 1x1024 column (row)
    so we can do linear regression on it
    
    id      - input data array, int the format
    source_folder - the folder of which files are stored

    '''
    log_event(" Collapsing Images ",2)
    i = 0
    x = empty((size(in_set),1024))
    for j in in_set:
        im = imread(source_folder + j)
        x[i] = im.flatten()
        log_event("  PASS  " + j,1)
        i+=1
    
    
    log_event(" Collapsing Images ",2)
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
    
    train_x = collapse_image(act_set[4][1],"resized_all/")          # Bill Hader
    train_x = np.append(train_x, collapse_image(act_set[5][1],"resized_all/"), axis=0)    # Steve Carell
    train_x = np.append(train_x, collapse_image(act_set[3][1],"resized_all/"), axis=0)    # Alec Baldwin
    train_x = np.append(train_x, collapse_image(act_set[2][1],"resized_all/"), axis=0)    # Kristin Chenoweth
    train_x = np.append(train_x, collapse_image(act_set[1][1],"resized_all/"), axis=0)    # America Ferrera
    train_x = np.append(train_x, collapse_image(act_set[0][1],"resized_all/"), axis=0)    # Fran Drescher
    
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
    
    train_x = collapse_image(act_set[4][2],"resized_all/")          # Bill Hader
    train_x = np.append(train_x, collapse_image(act_set[5][2],"resized_all/"), axis=0)    # Steve Carell
    train_x = np.append(train_x, collapse_image(act_set[3][2],"resized_all/"), axis=0)    # Alec Baldwin
    train_x = np.append(train_x, collapse_image(act_set[2][2],"resized_all/"), axis=0)    # Kristin Chenoweth
    train_x = np.append(train_x, collapse_image(act_set[1][2],"resized_all/"), axis=0)    # America Ferrera
    train_x = np.append(train_x, collapse_image(act_set[0][2],"resized_all/"), axis=0)    # Fran Drescher
    
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
    
    train_x = collapse_image(act_set[4][3],"resized_all/")          # Bill Hader
    train_x = np.append(train_x, collapse_image(act_set[5][3],"resized_all/"), axis=0)    # Steve Carell
    train_x = np.append(train_x, collapse_image(act_set[3][3],"resized_all/"), axis=0)    # Alec Baldwin
    train_x = np.append(train_x, collapse_image(act_set[2][3],"resized_all/"), axis=0)    # Kristin Chenoweth
    train_x = np.append(train_x, collapse_image(act_set[1][3],"resized_all/"), axis=0)    # America Ferrera
    train_x = np.append(train_x, collapse_image(act_set[0][3],"resized_all/"), axis=0)    # Fran Drescher
    
    return train_x, train_y.T
        
def p7_8(act_set, af, lam = 0.0, iter = 5000, nhid = 360):
    ''' This is the modifed guerzhoian function to nerual net
    
        :param act_set:  input actor set, in the format:    
                        [   [[actor_name],[Traning_set],[validation_set],[test_set]], 
                        [[actor_name],[Traning_set],[validation_set],[test_set]], 
                        [...] ]                        
        :param af:      activatoin function to use, "sigmoid" or "relu"
        :param lam:     regularization factor lambda
        :param iter:    iterations to run
        :param nhid:    number of hidden units
        
        
        :returns:   sess, W0, b0, W1, b1  these are the nerual net weights and session
                    to be used outside the funtion to calculate the net
    
    '''
    # Changes:
    #     input image size is 1024
    #     training everything in one batch since the training set is realtively small
        
        
    x = tf.placeholder(tf.float32, [None, 1024])
    
    
    #nhid = 50
    W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.0001, seed=411))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.0001, seed=411))
    
    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.0001))
    b1 = tf.Variable(tf.random_normal([6], stddev=0.0001))
    
    # snapshot = cPickle.load(open("snapshot50.pkl"))
    # W0 = tf.Variable(snapshot["W0"])
    # b0 = tf.Variable(snapshot["b0"])
    # W1 = tf.Variable(snapshot["W1"])
    # b1 = tf.Variable(snapshot["b1"])
    
    if af == "relu":
        layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)
    elif af == "sigmoid":
        layer1 = tf.nn.sigmoid(tf.matmul(x, W0)+b0)
    else:
        print("Bad activation functoin input, defaulting to sig")
        layer1 = tf.nn.sigmoid(tf.matmul(x, W0)+b0)
        

    layer2 = tf.matmul(layer1, W1)+b1
    
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])
    

    #lam = 0.0#10#0.0005
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty    
    train_step = tf.train.AdamOptimizer(5e-5).minimize(reg_NLL)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    train_x, train_y = get_train(act_set)
    vali_x, vali_y = get_vali(act_set)
    test_x, test_y = get_test(act_set)
    
    plot_x = []
    plot_test = []
    plot_train = []
    plot_vali = []
    
    for i in range(iter):
        #print i  
        #batch_xs, batch_ys = get_train_batch(M, 500)
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})
        
        if i % 1 == 0:
            plot_test.append(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
            plot_train.append(sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
            plot_vali.append(sess.run(accuracy, feed_dict={x: vali_x, y_: vali_y}))
            plot_x.append(i)
        
        if i % 100 == 0:
            print "i=",i
            print "Test:", plot_test[-1]
        
            print "Train:", plot_train[-1]
            print "Penalty:", sess.run(decay_penalty)
        
        
            snapshot = {}
            snapshot["W0"] = sess.run(W0)
            snapshot["W1"] = sess.run(W1)
            snapshot["b0"] = sess.run(b0)
            snapshot["b1"] = sess.run(b1)
            #cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))
            
    try:
        plt.figure()
        plt.plot(plot_x, plot_train, '-g', label='Training')
        plt.plot(plot_x, plot_vali, '-r', label='Validation')
        plt.plot(plot_x, plot_test, '-b', label='Test')
        plt.xlabel("Training Iterations")
        plt.ylabel("Accuracy")
        plt.title("Traning Curve for Single Hidden Layer NN, lam = {}".format(lam))
        plt.legend(loc='bottom right')
        plt.show()
    except:
        print("plot fail")
        
    print "Final test set accuracy:"
    print (sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    
    return sess, W0, b0, W1, b1
        
        
    def p9(act_set, actors):    
        
        # Train and save network from part 7
        sess, W0, b0, W1, b1 = p7_8(act_set, "sigmoid", 0.1, 2000, 360)
        
        for i in actors:
            # Loop through each actor
        
        
        

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
    
    #p7_8(act_set, lam = 0.0, iter = 5000, nhid = 360, af):
    
    
    p7_8(act_set, "sigmoid", 0.1, 2000, 360)   
    
    p7_8(act_set, "relu", 0.1, 3000, 360)       
    p7_8(act_set, "relu", 1, 3000, 360)          
    p7_8(act_set, "relu", 2, 3000, 360)      
    p7_8(act_set, "relu", 3, 3000, 360)     
    p7_8(act_set, "relu", 6.67, 3000, 360)     
    p7_8(act_set, "relu", 10, 3000, 360)  
    
    p9
    
        
    
    # for i in range(10):
    #     square_theta = np.reshape(W0.eval(sess).T[i],(32,32))    
    #     try:
    #         plt.figure()
    #         plt.imshow(square_theta, interpolation='sinc')
    #         plt.title("Theta #:" + str(i))
    #         plt.show()
    #     except:
    #         print("plot fail")
            
        
    # a = np.dot([0,1,0,0,0,0], W1.eval(sess).T)
    # b = W0.eval(sess).T[np.argmax(a)]
    # square_theta = np.reshape(b,(32,32))            
    # plt.figure()
    # plt.imshow(square_theta, interpolation='mitchell')
    # plt.title("Theta Test")
    # plt.show()
    #     
    
    