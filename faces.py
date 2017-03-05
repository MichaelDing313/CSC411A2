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


## INIT CODE
__name__ = "__init__"

if __name__ == "__init__":
    # Execution start here
    
    # Set log level, for convience, avaliable options are:
    # ALL, SHORT, INFO, WARNING, NONE
    log_level = "SHORT"
    
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
    
    #Build actor list
    act_1 =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    
    # end of init code
    
if __name__ == "__main__":
    
    #t = int(time.time())
    t = 1454219613
    print "t=", t
    random.seed(t)
    
    
    M = loadmat("mnist_all.mat")
    
    import tensorflow as tf
        
    x = tf.placeholder(tf.float32, [None, 1024])
    
    
    nhid = 300
    W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
    
    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))
    
    # snapshot = cPickle.load(open("snapshot50.pkl"))
    # W0 = tf.Variable(snapshot["W0"])
    # b0 = tf.Variable(snapshot["b0"])
    # W1 = tf.Variable(snapshot["W1"])
    # b1 = tf.Variable(snapshot["b1"])
    
    
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1
    
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    
    
    lam = 0.00000
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x, test_y = get_test(M)