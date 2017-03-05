## Initial Imports and things

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
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


## Guerzhoy's Image Import code
# Thank you prof Guerzhoy

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
                
def fetch_image_URL_mod(actor_list, output_folder, source_list,output_list):
    testfile = urllib.URLopener()            
    out_file = open(output_list,'w')
    
    #Note: you need to create the uncropped folder first in order 
    #for this to work

    log_event(" Fetching Images From Web ",2)
    for a in actor_list:
        name = a.split()[1].lower()
        i = 0
        for line in open(source_list):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], output_folder+filename), {}, 30)
                
                # If there is no file, i.e theimage was not obtained
                if not os.path.isfile(output_folder+filename):
                    log_event(" Skip - Timed Out ",3)
                    continue
                    
                # Check if file is a valid image, eliminate semi dead links (404s)
                
                if (imghdr.what(output_folder+filename) == None):
                    log_event(" Skip - Image Invalid ",3)
                    continue
    
                # Only run if file is valid
                
                # Output file with crop info on the side
                out_file.write(a + ' ' + filename + ' ' + line.split()[5] + ' ' + line.split()[4] +'\n')
                log_event("  PASS  " + filename,1)
                i += 1
                log_event(" Retrieved: " + filename,1)

    out_file.close()
    log_event(" Image Fetch Complete ",2)

## Data Organization
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

# Pick a small set of data from library, sampling
def pick_subset_density(density,seed,input,output):
    ''' 
    This function let you pick a random subset of data from the set of all possible pictures
    
    density -   from 0 to 1, the rough ratio of data you want to keep (0.1 means output will be about 10% of the total set)
    seed    -   random seed, use for consistancy of results
    input   -   input file name to be processes
    output  -   output file name to save result to
    '''
    np.random.seed(seed)
    out_file = open(output, 'wb');
    
    for line in open(input):
        if (np.random.uniform(low=0.0, high=1.0, size=None) < density):
            out_file.write(line)
        else:
            continue
    
# Copy data from uncropped library to working dir
def copy_file(act, origin_folder, dest_folder, ref_file):
    log_event(" Copying File ",2)
    for a in act:
        for line in open(ref_file):
            if a in line:
                filename = line.split()[2] # Extract file name to move
                #A version without timeout (uncomment in case you need to 
               
                shutil.copy(origin_folder + filename, dest_folder + filename)
                
                # If there is no file, i.e copy failed
                if not os.path.isfile(dest_folder + filename):
                    log_event("  FAIL  " + filename,3)
                    continue
                
                log_event("  PASS  " + filename,1)

    log_event(" Copying Complete ",2)

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
            output_set[i][1] = temp_list[:100]
            output_set[i][2] = temp_list[100:110]
            output_set[i][3] = temp_list[110:120]
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
    
    in_set      - input data set, in the format:
                    [[filename_1],[filename_2], ... ,[filename_n]]
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
    
def grad_descent(f, df, x, y, init_t, alpha, EPS):    
    log_event(" Staring Gradient Descent ",2)
    #EPS = 1e-11   #EPS = 10**(-11)
    prev_t = init_t - 10*EPS
    t = init_t.copy()
    x = vstack( (ones((1, x.shape[1])), x))
    max_iter = 500000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        z = alpha * df(x, y, t)
        #print(shape(z))
        t -= z
        if iter % 1000 == 0:            
            log_event("Iter " + str(iter),1)
            log_event(str(f(x, y, t)),2)
        iter += 1
    
    if norm(t - prev_t) <=  EPS:
        log_event("  PASS  Converge",2) 
         
    if iter >= max_iter:
        log_event("  PASS  Itermax",2)
        

    log_event(" Gradient Descent Finished ",2)
    
    return t

def test_result_bi(test_set, theta, ans, input_folder):
    '''
    This code runs through the input x and y and test the theta, returns
    percentage of the correct answer
    This function is designed for single value classifier result (not vector)
    
    test_set        - input set, a list of filenames to test, in the format:
                        [[filename_1],[filename_2], ... ,[filename_n]]
    theta           - input theta fromgradient descent
    ans             - input y, matrix corrosponding to the correct answer
    input_folder    - folder to pull files from 
    
    '''
    correct, wrong = 0, 0
    i = 0
    for j in test_set:
        #y = compute_hypothesis(theta, j, "resized_all/")
        im = imread(input_folder + j)
        x = np.append(1, im.flatten())
        y = dot(theta.T, x)
        if y > 0:
            y = 1
        else:
            y = -1
        
        if y == ans[i]:
            log_event("  Guess Correct  "+ str(y),1)
            correct += 1
        else:
            log_event("  Guess Wrong  "+ str(y),1)
            wrong += 1
        i += 1
    correct = float(correct)
    wrong = float(wrong)
    return (correct / (correct+wrong) * 100 )
    
def test_result_multi(test_set, theta, ans, input_folder):
    '''
    This code runs through the input x and y and test the theta, returns
    percentage of the correct answer
    This function is designed for single value classifier result (not vector)
    
    test_set        - input set, a list of filenames to test, in the format:
                        [[filename_1],[filename_2], ... ,[filename_n]]
    theta           - input theta fromgradient descent
    ans             - input y, matrix corrosponding to the correct answer
    input_folder    - folder to pull files from 
    
    '''
    correct, wrong = 0, 0
    i = 0
    for j in test_set:
        im = imread(input_folder + j)
        x = np.append(1, im.flatten())
        y = dot(theta.T, x)
        
        if np.argmax(y) == np.argmax(ans.T[i]):
            log_event("  Guess Correct  "+ str(y),1)
            correct += 1
        else:
            log_event("  Guess Wrong  "+ str(y),1)
            wrong += 1
        i += 1
    correct = float(correct)
    wrong = float(wrong)
    return (correct / (correct+wrong) * 100 )
    

## Misc functions
# Functions for life improvment, such as handle logging
def log_event(log_string, level):
    '''
    This function handles info display in console, mainly here to unify logging
    and let me control the level of logging displayed
    log_string  - Log message to be displayed
    level       - Severity of event, higher the number more important
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
        print '.',
    else:
        print(log_string)
    
    

## Main Function
# Execution start here

# Set log level, for convience, avaliable options are:
# ALL, SHORT, INFO, WARNING, NONE
log_level = "SHORT"

## Part 1 Gather Data
# Download, filter, crop and resize images to be used

#Build actor list
act_1 =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

#uncomment all these code below to run entire process
files = ["facescrub_actors.txt","facescrub_actresses.txt"]
newfile = open('combined_face_scrub.txt', 'w')
newfile.write( ''.join([open(f).read() for f in files]))
newfile.close

act = list(set([a.split("\t")[0] for a in open("combined_face_scrub.txt").readlines()]))
make_folders(('uncropped_all/','uncropped_all','cropped_all','grey_all','resized_all','set_2','pt8'))
fetch_image_URL_mod(act, "uncropped_all/","combined_face_scrub.txt","uncropped_all/0_subset_crop_info.txt")
crop_images(act, "uncropped_all/","cropped_all/","uncropped_all/0_subset_crop_info.txt","cropped_all/0_output_info.txt")
convert_grey(act, "cropped_all/","grey_all/","uncropped_all/0_subset_crop_info.txt")
resize_32x32(act, "grey_all/","resized_all/","cropped_all/0_output_info.txt","resized_all/0_set_info.txt")

# end of init code

## Part 2 Seperate data
# Pick the data set to be used for each actor

act_set = pick_sets(act, "resized_all/0_set_info.txt")
print('actor order are:')
actor_order = []
count = 0
for i in act_set:
    print(str(count) + ':' + i[0])
    actor_order.append(i[0])
    count += 1
print(actor_order)

## Part 3 Linear Regression on Two Actor
# Find and gradient descend cost function

# Define cost function here
# h_theta = dot(theta.T,x)

def f(x, y, theta):
    return sum( (y-dot(theta.T,x)) ** 2)

def df(x, y, theta):
    return -2*sum((y-dot(theta.T, x))*x, 1)


train_x = collapse_image(act_set[7][1],"resized_all/")          # Bill Hader
train_x = np.append(train_x, collapse_image(act_set[8][1],"resized_all/"), axis=0)    # Steve Carell
train_y = np.append(linspace(1,1, 100), linspace(-1,-1, 100),axis=0)           # Bill = 1, Steve = -1

theta0 = linspace(0.5,0.5, 1025)
theta = grad_descent(f, df, train_x.T/255, train_y, theta0, 1e-6,1e-8)

np.save("pt3_theta1_1.faceVar", theta)
#theta = np.load("pt3_theta1_1.faceVar.npy")

#check result on training set:
test_set = np.append(act_set[7][1],act_set[8][1])   #Construct List of files in the test
test_result = test_result_bi(test_set,theta,train_y,"resized_all/") #Test the set and save result
print("Test Result, Training Set:")     #Print result on screen
print(str(test_result)+" percent")
print("cost fcn: ", f(vstack( (ones((1, train_x.T.shape[1])), train_x.T/255)),train_y,theta))


train_x = collapse_image(act_set[7][2],"resized_all/")          # Bill Hader
train_x = np.append(train_x, collapse_image(act_set[8][2],"resized_all/"), axis=0)    # Steve Carell
#check result on validation set
vali_y = np.append(linspace(1,1, 10), linspace(-1,-1, 10),axis=0)           # Bill = 1, Steve = -1
test_set = np.append(act_set[7][2],act_set[8][2])
test_result = test_result_bi(test_set, theta,vali_y,"resized_all/")    
print("Test Result, Validation Set:")
print(str(test_result)+" percent")
print("cost fcn: ", f(vstack( (ones((1, train_x.T.shape[1])), train_x.T/255)),vali_y,theta))

train_x = collapse_image(act_set[7][3],"resized_all/")          # Bill Hader
train_x = np.append(train_x, collapse_image(act_set[8][3],"resized_all/"), axis=0)    # Steve Carell
#check result on validation set
test_y = np.append(linspace(1,1, 10), linspace(-1,-1, 10),axis=0)           # Bill = 1, Steve = -1
test_set = np.append(act_set[7][3],act_set[8][3])
test_result = test_result_bi(test_set, theta,test_y,"resized_all/")        
print("Test Result, Test Set:")
print(str(test_result)+" percent")
print("cost fcn: ", f(vstack( (ones((1, train_x.T.shape[1])), train_x.T/255)),vali_y,theta))

## part 4 Visualize Theta

# Take Theta and make back into 32x32
square_theta = np.reshape(theta[1:],(32,32))
# Plot to window and save
try:
    plt.imshow(square_theta, interpolation='nearest')
    plt.title("Theta, optimized for full 200 image training set")
    plt.show()
except:
    print("plot fail")
imsave("pt3_visual_theta.png", square_theta)

# Define new data set for the second case, and re-gradient descent
train_x = collapse_image([act_set[7][1][1]],"resized_all/")          # Bill Hader
train_x = np.append(train_x, collapse_image([act_set[8][1][1]],"resized_all/"), axis=0)    # Steve Carell
train_y = np.append(linspace(1,1, 1), linspace(-1,-1, 1),axis=0)           # Bill = 1, Steve = -1

theta0 = linspace(1,1, 1025)
theta = grad_descent(f, df, train_x.T/255, train_y, theta0, 1e-5,1e-3)
np.save("pt3_theta2.faceVar", theta)

# Make theta back to 32x32, open new figre, plot and save image
square_theta = np.reshape(theta[1:],(32,32))
try:
    plt.figure()
    plt.imshow(square_theta, interpolation='nearest')
    plt.title("Theta, optimized for one image from each actor")
    plt.show()
except:
    print("plot fail")
imsave("pt3_visual_theta2.png", square_theta)

## part 5 male/female overfitting

k = 0
size_range = linspace(1, 90, num=90).astype(int)
repeat_result = [None] * 91
plot_x = []
plot_vali = []
plot_test = []
# loop through a bunch of different data sizes and save result
# for grading purposes you probably want to shorten the input list
# replace above with, this still might take a bit especailyl for the last loop

#k = 0
#size_range = linspace(1, 60, num=60).astype(int)
#repeat_result = linspace(1, 60, num=60)

for k in size_range:
    # Define the x training set, combine all the image data
    train_x = collapse_image(act_set[7][1][0:k],"resized_all/")          # Bill Hader
    train_x = np.append(train_x, collapse_image(act_set[8][1][0:k],"resized_all/"), axis=0)    # Steve Carell
    train_x = np.append(train_x, collapse_image(act_set[6][1][0:k],"resized_all/"), axis=0)    # Alec Baldwin
    train_x = np.append(train_x, collapse_image(act_set[11][1][0:k],"resized_all/"), axis=0)    # Kristin Chenoweth
    train_x = np.append(train_x, collapse_image(act_set[9][1][0:k],"resized_all/"), axis=0)    # America Ferrera
    train_x = np.append(train_x, collapse_image(act_set[2][1][0:k],"resized_all/"), axis=0)    # Fran Drescher
    
    # Create answer data to train against
    train_y = np.append(linspace(1,1, 3*k), linspace(-1,-1, 3*k),axis=0)           # male = 1, female = -1
    
    # Initialize and run gradient descent given the data size
    theta0 = linspace(1,1, 1025)
    theta = grad_descent(f, df, train_x.T/255, train_y, theta0, 1e-6,1e-5)
    
    # Save or load theta, dont need to re calculate theta every time
    np.save("set_2/pt5_repeat_"+str(k), theta)
    #theta = np.load("set_2/pt5_repeat_"+str(k)+".npy")
    
    
    #check result on set of actors not trained for
    test_y = np.append(linspace(1,1, 300), linspace(-1,-1, 300),axis=0)           # male = 1, female = -1
    test_set = np.append(act_set[1][1],act_set[10][1])
    test_set = np.append(test_set,act_set[5][1])
    test_set = np.append(test_set,act_set[3][1])
    test_set = np.append(test_set,act_set[4][1])
    test_set = np.append(test_set,act_set[0][1])    
    test_result = test_result_bi(test_set, theta, test_y,"resized_all/")   
    
    # Build image input data for validation set, these actors were part of the training set
    vali_y = np.append(linspace(1,1, 30), linspace(-1,-1, 30),axis=0)    
    vali_set = act_set[7][2]          # Bill Hader
    vali_set = np.append(vali_set, act_set[8][2], axis=0)    # Steve Carell
    vali_set = np.append(vali_set, act_set[6][2], axis=0)    # Alec Baldwin
    vali_set = np.append(vali_set, act_set[11][2], axis=0)    # Kristin Chenoweth
    vali_set = np.append(vali_set,act_set[9][2], axis=0)    # America Ferrera
    vali_set = np.append(vali_set, act_set[2][2], axis=0)    # Fran Drescher
    vali_result = test_result_bi(vali_set, theta, vali_y,"resized_all/")    
    
    
    # Report on results of the test each loop
    print("Test Result, Gender:")
    print(str(test_result)+" percent")
    repeat_result[k] = [k,vali_result,test_result]
    
# Report on the final results obtained from gradient descend
for i in repeat_result[1:]:
    print("Training Size: %d , Validation: %d, Inference: %d",i[0]*6, i[1], i[2])
    plot_x.append(i[0]*6)
    plot_vali.append(i[1])
    plot_test.append(i[2])

try:
    plt.figure()
    plt.plot(plot_x, plot_vali, '-r', label='Validation')
    plt.plot(plot_x, plot_test, '-b', label='Untrained')
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title("Testing accuracy for various training set size")
    plt.legend(loc='bottom right')
    plt.show()
except:
    print("plot fail")


## part 6 multiclassification


def f(x, y, theta):
    return sum( (y-dot(theta.T,x)) ** 2)


def df(x, y, theta):
    # This is still vectoirzed and compute quickly, the steps are broken down
    # to allow better debugging outputs
    a = dot(theta.T, x)
    b = a-y
    c = 2*dot(x,b.T)
    return c



## Part 6d verify gradient using finite differences

h = 1e-5

#pick some random images to look at

x = collapse_image(act_set[7][3],"resized_all/").T
x = vstack( (ones((1, x.shape[1])), x))
y = linspace(1,1, 10)
theta0 = linspace(1,1, 1025).T
i = 0
while i < shape(x)[0]:
    if i%25 == 0:
        # test 12 components of the gradient function
        temp = theta0.copy()
        temp[i] -= h
        fake_grad = (f(x, y, theta0) - f(x, y, temp)) / h
        real_grad = df(x, y, theta0)
        print("Grad:" + str(real_grad[i]) + "   Error:"+ str((fake_grad - real_grad[i])))
    i += 1


## Part 7 run gradient descent for multi classification


act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

train_x = collapse_image(act_set[7][1],"resized_all/")          # Bill Hader
train_x = np.append(train_x, collapse_image(act_set[8][1],"resized_all/"), axis=0)    # Steve Carell
train_x = np.append(train_x, collapse_image(act_set[6][1],"resized_all/"), axis=0)    # Alec Baldwin
train_x = np.append(train_x, collapse_image(act_set[11][1],"resized_all/"), axis=0)    # Kristin Chenoweth
train_x = np.append(train_x, collapse_image(act_set[9][1],"resized_all/"), axis=0)    # America Ferrera
train_x = np.append(train_x, collapse_image(act_set[2][1],"resized_all/"), axis=0)    # Fran Drescher

train_y = np.append(linspace(1,1, 100), linspace(0,0, 500))                                      # bill = [1,0,0,0,0,0]
train_y = vstack( (np.append(np.append(linspace(0,0, 100), linspace(1,1, 100)), linspace(0,0, 400)), train_y))   # steve = [0,1,0,0,0,0]
train_y = vstack( (np.append(np.append(linspace(0,0, 200), linspace(1,1, 100)), linspace(0,0, 300)), train_y))    # alec = [0,0,1,0,0,0]
train_y = vstack( (np.append(np.append(linspace(0,0, 300), linspace(1,1, 100)), linspace(0,0, 200)), train_y))    # kristin = [0,0,0,1,0,0]
train_y = vstack( (np.append(np.append(linspace(0,0, 400), linspace(1,1, 100)), linspace(0,0, 100)), train_y))    # america = [0,0,0,0,1,0]
train_y = vstack( (np.append(linspace(0,0, 500), linspace(1,1, 100)), train_y))    # fran = [0,0,0,0,0,1]


theta0 = vstack((linspace(1,1, 1025),linspace(1,1, 1025),linspace(1,1, 1025),linspace(1,1, 1025),linspace(1,1, 1025),linspace(1,1, 1025))).T
theta = grad_descent(f, df, train_x.T/255, train_y, theta0,1e-6,1e-6)

np.save("pt6_theta2.faceVar", theta)

#theta = np.load("pt6_theta2.faceVar.npy")

#check result on training set:
test_set = np.append(act_set[7][1],[act_set[8][1],\
                    act_set[6][1],act_set[11][1],\
                    act_set[9][1],act_set[2][1]])   #Construct List of files in the test


test_result = test_result_multi(test_set,theta,train_y,"resized_all/") #Test the set and save result
print("Test Result, Training Set:")     #Print result on screen
print(str(test_result)+" percent")
print("cost fcn: ", f(vstack( (ones((1, train_x.T.shape[1])), train_x.T/255)),train_y,theta))

#check result on validation set:
test_set = np.append(act_set[7][2],[act_set[8][2],\
                    act_set[6][2],act_set[11][2],\
                    act_set[9][2],act_set[2][2]])  #Construct List of files in the test
                    
                    
# Construct image array for validation set for computing cost function
train_x = collapse_image(act_set[7][2],"resized_all/")          # Bill Hader
train_x = np.append(train_x, collapse_image(act_set[8][2],"resized_all/"), axis=0)    # Steve Carell
train_x = np.append(train_x, collapse_image(act_set[6][2],"resized_all/"), axis=0)    # Alec Baldwin
train_x = np.append(train_x, collapse_image(act_set[11][2],"resized_all/"), axis=0)    # Kristin Chenoweth
train_x = np.append(train_x, collapse_image(act_set[9][2],"resized_all/"), axis=0)    # America Ferrera
train_x = np.append(train_x, collapse_image(act_set[2][2],"resized_all/"), axis=0)    # Fran Drescher

#Construct List of answers
vali_y = np.append(linspace(1,1, 10), linspace(0,0, 50))                                      # bill = [1,0,0,0,0,0]
vali_y = vstack( (np.append(np.append(linspace(0,0, 10), linspace(1,1, 10)), linspace(0,0, 40)), vali_y))   # steve = [0,1,0,0,0,0]
vali_y = vstack( (np.append(np.append(linspace(0,0, 20), linspace(1,1, 10)), linspace(0,0, 30)), vali_y))    # alec = [0,0,1,0,0,0]
vali_y = vstack( (np.append(np.append(linspace(0,0, 30), linspace(1,1, 10)), linspace(0,0, 20)), vali_y))    # kristin = [0,0,0,1,0,0]
vali_y = vstack( (np.append(np.append(linspace(0,0, 40), linspace(1,1, 10)), linspace(0,0, 10)), vali_y))    # america = [0,0,0,0,1,0]
vali_y = vstack( (np.append(linspace(0,0, 50), linspace(1,1, 10)), vali_y))    # fran = [0,0,0,0,0,1]

test_result = test_result_multi(test_set,theta,vali_y,"resized_all/") #Test the set and save result
print("Test Result, Validation Set:")     #Print result on screen
print(str(test_result)+" percent")
print("cost fcn: ", f(vstack( (ones((1, train_x.T.shape[1])), train_x.T/255)),vali_y,theta))

#check result on test set:
test_set = np.append(act_set[7][3],[act_set[8][3],\
                    act_set[6][3],act_set[11][3],\
                    act_set[9][3],act_set[2][3]])   #Construct List of files in the test
                    
                    
# Construct image array for validation set for computing cost function
train_x = collapse_image(act_set[7][3],"resized_all/")          # Bill Hader
train_x = np.append(train_x, collapse_image(act_set[8][3],"resized_all/"), axis=0)    # Steve Carell
train_x = np.append(train_x, collapse_image(act_set[6][3],"resized_all/"), axis=0)    # Alec Baldwin
train_x = np.append(train_x, collapse_image(act_set[11][3],"resized_all/"), axis=0)    # Kristin Chenoweth
train_x = np.append(train_x, collapse_image(act_set[9][3],"resized_all/"), axis=0)    # America Ferrera
train_x = np.append(train_x, collapse_image(act_set[2][3],"resized_all/"), axis=0)    # Fran Drescher
                    
#Use the same answer data from before, since they are the same

test_result = test_result_multi(test_set,theta,vali_y,"resized_all/") #Test the set and save result
print("Test Result, test Set:")     #Print result on screen
print(str(test_result)+" percent")
print("cost fcn: ", f(vstack( (ones((1, train_x.T.shape[1])), train_x.T/255)),vali_y,theta))

## Part 8 Visualize Theta
# Make theta back to 32x32, open new figre, plot and save image
i = 0
while i < 6:
    square_theta = np.reshape(theta.T[i][1:],(32,32))
    try:
        plt.figure()
        plt.imshow(square_theta, interpolation='nearest')
        plt.title("Theta #:" + str(i))
        plt.show()
    except:
        print("plot fail")
    imsave("./pt8/pt8_visual_theta" + str(i) + ".png", square_theta)
    i += 1

print("Done!")
print("Done!")
print("Done!")
print("Done!")
print("Done!")
print("Done!")