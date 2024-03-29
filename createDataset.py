import cv2
import numpy as np
import operator
from matplotlib import pyplot as plt
from scipy import signal as sig
import scipy
import glob, os
import math
MIN_CONTOUR_AREA = 50
RESIZED_IMAGE_WIDTH = 40
RESIZED_IMAGE_HEIGHT = 50

class ContourWithData():
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX - 2
        self.intRectY = intY - 2
        self.intRectWidth = intWidth + 4
        self.intRectHeight = intHeight + 4

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

allContoursWithData = []                # declare empty lists,
validContoursWithData = []              # we will fill these shortly
imgTrainCharacters = cv2.imread("C:\\Users\\akbul\\Desktop\\ISSD STAJ\\OCR\\textIng.jpg")          # read in testing numbers image
freshImage=cv2.imread("C:\\Users\\akbul\\Desktop\\ISSD STAJ\\OCR\\text4.jpg")
imgGray = cv2.cvtColor(imgTrainCharacters, cv2.COLOR_BGR2GRAY)       # get grayscale image
imgBlurred = cv2.bilateralFilter(imgGray,5,75,75)               # bilateral filter
# filter image from grayscale to black and white
imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

med_th = scipy.signal.medfilt2d(imgThresh,5)#Remove Salt and Pepper Noise from Image
imgThreshCopy = med_th.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points
#take all character with contour area and corner points
for npaContour in npaContours:                            
    contourWithData = ContourWithData()                                             # instantiate a contour with data object
    contourWithData.npaContour = npaContour                                         # assign contour to contour with data
    contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
    contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
    contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
    allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data

#take just area bigger than MIN_CONTOUR_AREA
for contourWithData in allContoursWithData:                 # for all contours
    if contourWithData.checkIfContourIsValid():             # check if valid
        validContoursWithData.append(contourWithData)       # if so, append to valid contour list


#Draw rectangle to valid characters, crop this charcter from image and  resize this croped images, save this resize image into file with new name
i=1
for contourWithData in validContoursWithData:            # for each contour
                                            # draw a blue rect around the current char
    cv2.rectangle(imgTrainCharacters,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 0, 255),              # blue
                      1)                        # thickness
    crop_img_contours = med_th[contourWithData.intRectY:contourWithData.intRectY + contourWithData.intRectHeight, contourWithData.intRectX:contourWithData.intRectX + contourWithData.intRectWidth]
    imgResize=cv2.resize(crop_img_contours,(RESIZED_IMAGE_WIDTH , RESIZED_IMAGE_HEIGHT))
    savedımage=cv2.imwrite( 'C:\\Users\\akbul\\Desktop\\ISSD STAJ\\OCR\\letters\\'+str(i)+ '.jpg', imgResize)
    i=i+1
    
cv2.imshow('',imgTrainCharacters)
cv2.waitKey(0)                                          # wait for user key press
cv2.destroyAllWindows()             # remove windows from memory
