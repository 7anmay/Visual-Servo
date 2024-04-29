import cv2
import numpy as np
from conf import load_config


class ImageProcessor:

class CropLaneDetector:
    def __init__(self ) -> None:
        self.config = load_config()
        ###### initialise all parameters here ######

    def reset(self):
        ###### reset all parameters here ######
        pass

    def findCropLane(self, image):
        # call Process image to segment image
        # call findCropRows to find crop lane
        # return crop lane found bool

        return croplane_found
    
    def findCropRows(self, image):
        # find crop lane using troughs and peaks algorithm
        # save found lanes in class variables
        # return lines and linesROIs

        ## findlinesInImage()
        ## movingstd()
        ## findpeakstroughs()
        ##findcroprowsinMVSignal()
        
    def findLinesInImage(self):
        ## find line and window intersection points
        ## update trackingboxes
        ## upate plantsuIncroprow
        ## for more than 2 plants in row
        ## calc line angle
        ## update trackingbox width
        ## return lines, trackingWindows, meanLinesInWindows

    def updateTrackingBoxes(self):
        ## update tracking boxes
        ## return trackingBoxes
        ## acc to num_scansteps append tracking boxes initially
        ## once lines dtected update tracking boxes wrt line

    def checkPlantsInRows(self, cropRowID, plantsInCropRow):
        if len(plantsInCropRow) >= 2:
            return True
        else:
            self.lostCropRows[cropRowID] = 1
            return False        

    def checkPlantsLocTB(self, point):
        if point[1] < self.imgHeight/2:
            self.pointsInTop += 1
        else:
            self.pointsInBottom += 1

    def trackCropLane(self):
        

if __name__ == "__main__":
    main("./visualServo.png")