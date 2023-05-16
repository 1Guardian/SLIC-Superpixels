#==============================================================================
#
# Class : CS 6420
#
# Author : Tyler Martin
#
# Project Name : Project 5 | SLIC superpixels
# Date: 4/3/2023
#
# Description: This project implements a SLIC clustering method
#
# Notes: Since I know you prefer to read and work in C++, this file is set
#        up to mimic a standard C/C++ flow style, including a __main__()
#        declaration for ease of viewing. Also, while semi-colons are not 
#        required in python, they can be placed at the end of lines anyway, 
#        usually to denote a specific thing. In my case, they denote globals, 
#        and global access, just to once again make it easier to parse my code
#        and see what it is doing and accessing.
#
#==============================================================================

#"header" file imports
from imports import *
from checkImages import *
from grayScaleImage import *
from saveImage import *

#================================================================
#
# NOTES: THE OUTPATH WILL HAVE THE LAST / REMOVED IF IT EXISTS
#        THE imageType WILL HAVE A . APPLIED TO THE FRONT AFTER
#        CHECKING VALIDITY
#
#================================================================

rect_selector = None
fequency_representation_shifted = None
fig = None
filename = None
image = None
magnitude_spectrum = None
freq_filt_img = None
reset = False
sized = .50
label = "NIL"
center = "NIL"
ruler = 10
size = 10
title_window = 'SLIC'
cycles = 4
slicVal = cv2.ximgproc.SLICO
skele = 1
factor = 1


#================================================================
#
# Function: trackbar callbacks
#
# Description: These functions serve as a simple callback to 
#              the sliders and update their corresponding values
#
# Returns: null
#
#================================================================
def slic_trackbar(val):
    global slicVal
    if (val == 0):
        slicVal = cv2.ximgproc.SLIC
    if (val == 1):
        slicVal = cv2.ximgproc.SLICO
    if (val == 2):
        slicVal = cv2.ximgproc.MSLIC

def ruler_trackbar(val):
    global ruler
    ruler = val

def size_trackbar(val):
    global size
    size = val

def itter_trackbar(val):
    global cycles
    cycles = val

def skele_trackbar(val):
    global skele
    skele = val

#================================================================
#
# Function: button(num, nul)
#
# Description: This function is called to apply the SLIC operation
#              on the given image and display it. 
#
# Returns: null
#
#================================================================

def button(num, nul):
    global image
    global title_window
    global ruler
    global size
    global cycles
    global slicVal
    global skele
    global factor

    if ruler == size == 0:
        cv2.imshow(title_window, image)
        return

    h, w, _ = image.shape

    # gaussian blur
    src = cv2.GaussianBlur(image,(5,5),0)

    # SLIC
    slic = cv2.ximgproc.createSuperpixelSLIC(src,algorithm = slicVal, region_size = int(size), ruler = int(ruler))
    slic.iterate(cycles)

    #get stuff from slic obj
    labels = slic.getLabels()
    num_slic = slic.getNumberOfSuperpixels()
    mask = slic.getLabelContourMask(src,True)

    cluster_sums = np.zeros((num_slic,3))
    cluster_sums_size = np.zeros(num_slic)

    #two pass- slow mean
    for x in range(int(h)):
        for y in range(int(w)):
            cluster_sums[labels[x][y]][0] += src[x][y][0]
            cluster_sums[labels[x][y]][1] += src[x][y][1]
            cluster_sums[labels[x][y]][2] += src[x][y][2]

            cluster_sums_size[labels[x][y]] += 1

    for x in range(num_slic):
        cluster_sums[x][0] = cluster_sums[x][0] / cluster_sums_size[x]
        cluster_sums[x][1] = cluster_sums[x][1] / cluster_sums_size[x]
        cluster_sums[x][2] = cluster_sums[x][2] / cluster_sums_size[x]

    for x in range(int(h)):
        for y in range(int(w)):
            src[x][y][0] = cluster_sums[labels[x][y]][0]
            src[x][y][1] = cluster_sums[labels[x][y]][1]
            src[x][y][2] = cluster_sums[labels[x][y]][2]

    mask = (mask + 1) * 255

    if (skele):
        src = cv2.bitwise_and(src, src, mask = mask)

    cv2.imshow(title_window, src)

#================================================================
#
# Function: __main__
#
# Description: This function is the python equivalent to a main
#              function in C/C++ (added just for ease of your
#              reading, it has no practical purpose)
#
#================================================================

def __main__(argv):

    #globals
    global rect_selector
    global fequency_representation_shifted
    global fig
    global filename
    global image
    global magnitude_spectrum
    global factor

    #variables that contain the command line switch
    #information
    inPath = "nothing"
    depth = 1
    mode = 1
    intensity = 1
    primary = "nothing"
    filename = "outImage"
    direction = 0

    # get arguments and parse
    try:
      opts, args = getopt.getopt(argv,"h:t:s:")

    except getopt.GetoptError:
            print("slic [-h] -t input_image -s scaling_factor")
            print("===========================================================================================================")
            print("-t : Target Image (t)")
            print("-s : Image Scaling Factor (s)")
            sys.exit(2)
    for opt, arg in opts:

        if opt == ("-h"):
            print("slic [-h] -t input_image -s scaling_factor")
            print("===========================================================================================================")
            print("-t : Target Image (t)")
            print("-s : Image Scaling Factor (s)")

        elif opt == ("-m"):
            if (int(arg) < 10 and int(arg) > 0):
                mode = int(arg)
            else:
                print("Invalid Mode Supplied. Only Values 1 through 9 Are Accepted.")

        elif opt == ("-t"):
            primary = arg
        elif opt == ("-s"):
            factor = int(arg)

    #demand images if we are not supplied any
    if (primary == "nothing"):
        print("slic [-h] -t input_image -s scaling_factor")
        print("===========================================================================================================")
        print("-t : Target Image (t)")
        print("-s : Image Scaling Factor (s)")
        sys.exit(2)

    

    #open the image
    image = checkImages(primary)
    h, w, _ = image.shape
    image = cv2.resize(image, (int(w/factor),int(h/factor)))

    cv2.namedWindow(title_window)

    cv2.createTrackbar('Ruler', title_window , 10, 100, ruler_trackbar)
    cv2.createTrackbar('Size', title_window , 10, 100, size_trackbar)
    cv2.createTrackbar('Iterations', title_window , 4, 10, itter_trackbar)
    cv2.createTrackbar('[0 == SLIC | 1 == SLICO | 2 == MSLIC]', title_window , 1, 2, slic_trackbar)
    cv2.createTrackbar('Show Bounaries', title_window , 1, 1, skele_trackbar)
    cv2.createButton('Apply SLIC', button, 0)

    cv2.imshow(title_window, image)
    cv2.waitKey()

#start main
argv = ""
__main__(sys.argv[1:])
