# TrainAndTest.py

import cv2
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
from PIL import Image


size = (1790,2504)
i=Image.open("form.jpg")
i.thumbnail(size)
i.save('form.jpg')


img = Image.open("form.jpg")
img1 = img.crop((551, 689, 1619, 730))
img1.save("Generated_assets/accno.jpg")
img2 = img.crop((644, 1032, 944, 1115))
img2.save("Generated_assets/sig1.jpg")
img3 = img.crop((973, 1032, 1273, 1115))
img3.save("Generated_assets/sig2.jpg")
img4 = img.crop((1318, 1032, 1618, 1115))
img4.save("Generated_assets/sig3.jpg")
img5 = img.crop((1381, 541, 1428, 575))
img5.save("Generated_assets/date-d.jpg")
img6 = img.crop((1444, 542, 1493, 573))
img6.save("Generated_assets/date-m.jpg")
img5 = img.crop((1508, 544, 1617, 575))
img5.save("Generated_assets/date-y.jpg")
img5 = img.crop((303, 1039, 362, 1078))
img5.save("Generated_assets/from-d.jpg")
img5 = img.crop((383, 1039, 441, 1078))
img5.save("Generated_assets/from-m.jpg")
img5 = img.crop((460, 1037, 593, 1078))
img5.save("Generated_assets/from-y.jpg")
img5 = img.crop((300, 1088, 363, 1123))
img5.save("Generated_assets/to-d.jpg")
img5 = img.crop((381, 1089, 442, 1122))
img5.save("Generated_assets/to-m.jpg")
img5 = img.crop((460, 1090, 593, 1122))
img5.save("Generated_assets/to-y.jpg")


# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

###################################################################################################
def scan(String):
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("Assets/classifications.txt", np.float32)                  # read in training classifications
    except:
        print "error, unable to open Assets/classifications.txt, exiting program\n"
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("Assets/flattened_images.txt", np.float32)                 # read in training images
    except:
        print "error, unable to open Assets/flattened_images.txt, exiting program\n"
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = cv2.imread(String)          # read in testing numbers image

    if imgTestingNumbers is None:                           # if image was not read successfully
        print "error: image not read from file \n\n"        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:            # for each contour
                                                # draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

        strFinalString = strFinalString + strCurrentChar            # append current char to full string
    # end for

    print "\n" + strFinalString + "\n"                  # show the full string

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # show input image with green boxes drawn around found digits
    cv2.waitKey(0)                                          # wait for user key press

    cv2.destroyAllWindows()             # remove windows from memory
    return strFinalString



def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_img(imageA, imageB):
        m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
	a=s*100
	p=('The Confidence Score between two images is %f percent\n'%a)
	return p


def compare_img1(imageA, imageB):
        m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
	a=s*100
	return a
        

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
	fig = plt.figure(title)
	plt.suptitle(" SSIM: %.2f" % (s))
	a=s*100
	p=('The similarity between two images is %f percent'%a)
	print(p)
        
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	plt.show()
	return p


print("Account number - ")
txt = ("Generated_assets/accno.jpg")
accno = scan(txt)
print("Date on which form was filled")
txt = ("Generated_assets/date-d.jpg")
dated = scan(txt)
txt = ("Generated_assets/date-m.jpg")
datem = scan(txt)
txt = ("Generated_assets/date-y.jpg")
datey = scan(txt)
print("Mandate Start Date")
txt = ("Generated_assets/from-d.jpg")
fromd = scan(txt)
txt = ("Generated_assets/from-m.jpg")
fromm = scan(txt)
txt = ("Generated_assets/from-y.jpg")
fromy = scan(txt)
print("Mandate End Date")
txt = ("Generated_assets/to-d.jpg")
tod = scan(txt)
txt = ("Generated_assets/to-m.jpg")
tom = scan(txt)
txt = ("Generated_assets/to-y.jpg")
toy = scan(txt)
with open('Results.txt', 'w') as file:
    ab = ("Account number - "+accno+ "\n")
    file.write(ab)
    ab = ("Date - " + dated + "/" + datem + "/" + datey + "\n")
    file.write(ab)
    ab = ("Mandate Start Date - " + fromd + "/" + fromm + "/" + fromy + "\n")
    file.write(ab)
    ab = ("Mandate End Date - " + tod + "/" + tom + "/" + toy + "\n")
    file.write(ab)

number = 3

c=('Original_Signature.jpg')
original= cv2.imread(c)
        
sample = []
for x in range(number):
        f=x+1
        c=('Generated_assets/sig%d.jpg'%(f))
        e= cv2.imread(c)
        sample.append(e)



# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
for x in range(number):
        sample[x] = cv2.cvtColor(sample[x], cv2.COLOR_BGR2GRAY)
        



sum1 = 0

# compare the images
for x in range(number):
        d=('Original vs sample[%d]'%x)
        compare_images(original, sample[x], d)

with open('Results.txt', 'a') as file:
        for x in range(number):
                ab = compare_img(original, sample[x])
                file.write(ab)
                d = compare_img1(original, sample[x])
                sum1 = sum1 + d

sum1 = sum1/3

p=('The Final Confidence Score between two images is %f percent\n'%sum1)

with open('Results.txt', 'a') as file:
        file.write(p)
    
      
    










