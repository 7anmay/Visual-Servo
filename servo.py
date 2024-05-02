import cv2
import numpy as np
from conf.config import load_config 
from scipy.signal import find_peaks
import utils
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self, imagePath, type) -> None:
        self.config = load_config()
        self.image = cv2.imread(imagePath)
        self.image = cv2.resize(self.image, (1280, 720))
        h,w,channels = self.image.shape
        ###### initialise all parameters here ######
        self.type = type
        self.contourParams = self.config["contour"]
        self.roiParams = self.config["roi"]

        self.split = self.contourParams["split"]
        self.roi_vertices = [
                            (w/5, h/2),
                            (0, h),
                            (w, h),
                            (4*w/5, h/2),
                            (w / 3, h / 2),
                        ]
        self.boxes = []
    
    def processImage(self, image, type):
        image = image.astype(np.uint8)
        #print("DEBUG: image shape ", image.shape)
        #image = self.applyROI(image)
        image = self.crop_roi(image, np.array([self.roi_vertices], np.int32))

        self.image = image
        if type == "vanilla":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            a_channel = lab[:,:,1]
            th = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            roi_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            final_mask = cv2.bitwise_and(roi_gray, roi_gray, mask=th)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            a_channel = lab[:,:,1]
            crop_mask = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            kernel = np.ones((5,5),np.uint8)
            eroded = cv2.morphologyEx(crop_mask, cv2.MORPH_OPEN, kernel)
            final_mask = cv2.dilate(eroded, kernel, iterations=1)

        #print("DEBUG: final_mask shape ", final_mask.shape)
        cv2.imshow("masked Image", final_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        plantContours = self.getPlantContours(final_mask, self.contourParams["minArea"])
        self.boxes = self.boxFitting(plantContours)
        plantCenters = self.getContCenter(plantContours)
        # Draw contours on the image
        for box in self.boxes:
            cv2.drawContours(self.image, [box], -1, (0, 255, 0), 2)
        contour_image = cv2.drawContours(self.image, plantContours, -1, (0, 255, 0), 2)

        # Plot the centers of the plant contours
        for center in plantCenters:
            cv2.circle(contour_image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

        # Display the image with contours and centers
        cv2.imshow("Contours and Centers", contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        x=plantCenters[:,1]
        y=plantCenters[:,0]
        return final_mask, plantContours, np.array([x,y])
    
    def crop_roi(self,img,vertices):
        mask = np.zeros_like(img)
        match_mask_color = (255,) * 3
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    
    #Todo: fix apply roi
    def applyROI(self, img):
        _img = img.copy()
        self.img = _img
        h,w,channels = _img.shape
        # defining ROI windown on the image
        if self.roiParams["enable_roi"]:
            roi_vertices = [
                            (w/5, h/2),
                            (0, h),
                            (w, h),
                            (4*w/5, h/2),
                            (w / 3, h / 2),
                        ]
            mask = np.zeros_like(_img)
            match_mask_color = (255,) * channels
            cv2.fillPoly(_img, np.array([roi_vertices], np.int32), match_mask_color)
            masked_image = cv2.bitwise_and(_img, mask)
        return masked_image

    def boxFitting(self, contours):
        boxes = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            boxes.append(box)
        return boxes

    def getPlantContours(self, mask, minArea=100.0):
        cont = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        filter_conts = list()
        for i in range(len(cont)):
            if cont[i] is not None:
                if cv2.contourArea(cont[i])>minArea:
                    # print(cv2.contourArea(cont[i]))
                    if self.split:
                        cn_x, cn_y, cnt_w, cn_h = cv2.boundingRect(cont[i])
                        sub_contours = self.splitContours(cont[i], cn_x, cn_y, cnt_w, cn_h, 20)
                        for j in sub_contours:
                            if j != []:
                                filter_conts.append(j)
                    else:

                        filter_conts.append(cont[i])
        return filter_conts

    def getContCenter(self, conts):
        centerpts = np.zeros((len(conts),2))
        for i in range(len(conts)):
            M = cv2.moments(conts[i])
            if(M['m00']!=0):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centerpts[i,:] = [cx,cy]
        centerpts = centerpts[~np.all(centerpts == 0, axis=1)]
        return centerpts

    def splitContours(self, contour, x, y, w, h, max_coutour_height):
        """splits larg contours in smaller regions 

        Args:
            contour (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            w (_type_): _description_
            h (_type_): _description_

        Returns:
            _type_: sub polygons (seperated countours)
        """
        sub_polygon_num = h // max_coutour_height
        sub_polys = list()
        subContour = list()
        vtx_idx = list()
        #print("DEBUG in split contour", contour)
        contour = [contour.squeeze().tolist()]
        for subPoly in range(1, sub_polygon_num + 1):
            for vtx in range(len(contour[0])): 
                if  (subPoly - 1 * max_coutour_height) -1 <=  contour[0][vtx][1] and \
                    (subPoly * max_coutour_height) -1 >= contour[0][vtx][1] and \
                    vtx not in vtx_idx:
                    subContour.append([contour[0][vtx]])
                    vtx_idx.append(vtx)

            sub_polys.append(np.array(subContour))
            subContour = list()

        return sub_polys

class CropLaneDetector:
    
    def __init__(self) -> None:
        self.config = load_config()
        self.type = self.config["perception"]["type"]
        self.imagePath = self.config["perception"]["image_path"] 
        ###### initialise all parameters here ######
        self.scannerParams = self.config['scanner']
        self.trackerParams = self.config['tracker']
        self.imageProcessor =  ImageProcessor(self.imagePath, self.type)
        self.reset()

    def reset(self):
        ###### reset all parameters here ######
        self.count = 0
        self.pointsInTop = 0
        self.pointsInBottom = 0

        self.cropRows = []
        self.primaryImg = []
        self.graphicsImg = []
        self.cropLaneFound = False
        self.isInitialized = False
        self.cropRowEnd = False

        self.trackingBoxLoc = []

                # window parameters
        self.trackingWindowOffsTop = 0
        self.trackingWindowOffsBottom = 0
        self.trackingWindowTopScaleRatio = 0.4  # scales the top of the window
        self.imgHeight, self.imgWidth = 720, 1280
                
                # steps create linspace
        self.scanFootSteps = np.linspace(self.scannerParams["scanStartPoint"],
                                            self.scannerParams["scanEndPoint"],
                                            self.scannerParams["scanWindowWidth"])
        self.rowTrackingBoxes = []
        self.updateTrackingBoxes()

        self.numOfCropRows = len(self.rowTrackingBoxes)

    def findCropLane(self):
        # call Process image to segment image

        self.primaryImg = self.imageProcessor.image.copy()
        self.imgHeight, selfimgWidth, self.imgChannels = self.imageProcessor.image.shape
        print("DEBUG: H W ", self.imgHeight, self.imgWidth)
        cv2.imshow("Primary Image", self.primaryImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if self.primaryImg is None:
            print("Error: Image not found")
        else:
            self.mask, self.plantObjects, self.plantCenters = self.imageProcessor.processImage(self.primaryImg, self.type)
            self.primaryImg = self.imageProcessor.image.copy()
            #print("DEBUG: plantCenters shape", len(self.plantCenters[0]))
            self.numPlantsInScene = len(self.plantCenters[0])
        if not self.isInitialized:
            print("INFO: Finding crop lane")
            self.lines, self.linesroi  =self.findCropRows(self.primaryImg)
            self.drawGraphics()
        return self.cropLaneFound
    

    def findCropRows(self, image):
        # find crop lane using troughs and peaks algorithm
        # save found lanes in class variables
        # return lines and linesROIs

        ## findlinesInImage()
        ## movingstd()
        ## findpeakstroughs()
        ##findcroprowsinMVSignal()

        lines, linesROIs, _ = self.findLinesInImage()
        #print("DEBUG: lines shape", lines.shape)
        #print("DEBUG: linesROIs shape", linesROIs.shape)
        # print("DEBUG: linesROIs", linesROIs)
        if len(lines) != 0:
            # print("DEBUG: lines found", len(lines)) 
            # find the moving std deviation of the lines
            mvSingal = self.movingstd(lines[:,0])
            # find the peaks and the troughs of the signal
            peaks, troughs = self.findPeaksTroughs(mvSingal,0.5)
            self.CropRows, self.trackingBoxLogic = self.findCropRowsInMVSignal(peaks, troughs, mvSingal, lines, linesROIs)
            self.numOfCropRows = len(self.cropRows)
            self.listCropRows = list(np.zeros(self.numOfCropRows))
            if self.numOfCropRows != 0:
                self.isInitialized = True
                self.cropLaneFound = True
                print("[Info] initialized controller", len(self.cropRows), 'Window positions: ', self.CropRows[:,0].tolist())
            else:
                print("[Error] No crop rows found")
                self.cropLaneFound = False
        else:
            print("[Error] No lines found")
            self.cropLaneFound = False
        
        return lines, linesROIs
                 

    def findCropRowsInMVSignal(self, peaks, troughs, mvSignal, lines, linesROIs):
        if len(peaks) != 0 and len(troughs) !=0 and len(mvSignal) !=0:
            qualifiedLines = np.zeros((len(peaks)+1,2))
            windowLocations = np.zeros((len(peaks)+1,1))
            try:
                print("IN try of find crop rows in mv signal")
                idx = 0
                for k in range(0, len(peaks)+1):
                    if k ==0:
                        troughsLine = troughs[troughs < peaks[k]]
                    else:
                        if k<len(peaks):
                            tmp = troughs[troughs < peaks[k]]
                            troughsLine = tmp[tmp > peaks[k-1]]
                        else:
                            troughsLine = troughs[troughs > peaks[k-1]]
                        
                    if len(troughsLine) != 0:
                        bestLine = np.where((mvSignal[troughsLine] == np.min(mvSignal[troughsLine])))[0][0]
                        qualifiedLines[idx,:] = lines[troughsLine[bestLine]]

                        if linesROIs[troughsLine[bestLine]] != [0]:
                            windowLocations[idx,:] = linesROIs[troughsLine[bestLine]]
                            idx += 1
            except:
                ## take all nagative peaks -- fallback
                print("IN Except of find crop rows in mv signal")
                qualifiedLines = lines[troughs]
                windowLocations = np.mean(linesROIs[troughs])

        elif len(peaks) == 0 and len(troughs) !=0 and len(mvSignal)!=0:
            troughsLine = troughs
            bestLine = np.where(mvSignal[troughsLine] == np.min(mvSignal[troughsLine]))[0][0]

            qualifiedLines = np.zeros((1,2))
            windowLocations = np.zeros((1,2))

            qualifiedLines[0, :] = lines[troughsLine[bestLine]]
            windowLocations[0] = linesROIs[troughsLine[bestLine]]
        else:
            qualifiedLines = []
            windowLocations = []

        qualifiedLines = qualifiedLines[~np.all(qualifiedLines == 0, axis=1)]
        windowLocations = windowLocations[~np.all(windowLocations == 0, axis=1)]

        return qualifiedLines, windowLocations


    def findLinesInImage(self):
        ## find line and window intersection points
        ## update trackingboxes
        ## upate plantsIncroprow
        ## for more than 2 plants in row
        ## calc line angle
        ## update trackingbox width
        ## return lines, trackingWindows, meanLinesInWindows
        lines = np.zeros((len(self.scanFootSteps), 2))
        trackingWindows = np.zeros((len(self.scanFootSteps),1))
        meanLlinesInWindows = np.zeros((len(self.scanFootSteps),1))
        angleVar = None

        self.poinsInBottom = 0
        self.pointsInTop = 0

        for boxIdx in range(self.numOfCropRows):
            if self.isInitialized:
                lineIntersection = utils.lineIntersectionWin(self.CropRows[boxIdx, 1],
                                                    self.CropRows[boxIdx, 0],
                                                    self.imgHeight,
                                                    self.trackerParams["topOffset"],
                                                    self.trackerParams["bottomOffset"])
                self.updateTrackingBoxes(boxIdx, lineIntersection)
        
            plantsInCropRow = list()
            # print("DEBUG: nu of mplants in the scene", self.numPlantsInScene)
            for ptIdx in range(self.numPlantsInScene):
                self.checkPlantsLocTB(self.plantCenters[:, ptIdx])

                if self.rowTrackingBoxes[boxIdx].contains(Point(self.plantCenters[0, ptIdx], self.plantCenters[1, ptIdx])):
                    plantsInCropRow.append(self.plantCenters[:, ptIdx])
            # print("DEBUG: plantcenters", self.plantCenters.shape)
            if len(plantsInCropRow) >=2:
                # print("DEBUG: plantsInCropRow ", len(plantsInCropRow))
                ptsFlip = np.flip(plantsInCropRow, axis=1)
                # print("DEBUG: ptsFlip ", ptsFlip.shape)
                xM, xB = utils.getLineRphi(ptsFlip)
                t_i, b_i = utils.lineIntersectImgUpDown(xM, xB, self.imgHeight)
                l_i, r_i = utils.lineIntersectImgSides(xM, xB, self.imgWidth)
                #print("DEBUG: row ID:", boxIdx, t_i, b_i, l_i, r_i )
                if xM is not None and b_i >= 0 and b_i <= self.imgWidth:
                    lines[boxIdx,:] = [xB, xM]
                    if self.isInitialized == False:
                        trackingWindows[boxIdx] = self.scanFootSteps[boxIdx]
                        meanLlinesInWindows[boxIdx] = np.median(plantsInCropRow, axis=0)[0]
        
        if self.isInitialized and angleVar is not None:
            self.trackerParams["trackingBoxWidth"] = max(3*angleVar, self.trackerparams["trackingBoxWidth"])

        lines = lines[~np.all(lines == 0, axis=1)]
        trackingWindows = trackingWindows[~np.all(trackingWindows == 0, axis=1)]
        meanLlinesInWindows = meanLlinesInWindows[~np.all(meanLlinesInWindows == 0, axis=1)]

        return lines, trackingWindows, meanLlinesInWindows

        
    def updateTrackingBoxes(self, boxIdx=0, lineIntersection=None):
        ## update tracking boxes
        ## return trackingBoxes
        ## acc to num_scansteps append tracking boxes initially
        ## once lines dtected update tracking boxes wrt line
        
        if not self.isInitialized and lineIntersection==None:
            # initial tracking Boxes 
            for i in range(len(self.scanFootSteps)):        
                boxBL_x = self.scanFootSteps[i]
                boxBR_x = self.scanFootSteps[i] + self.trackerParams["trackingBoxWidth"]
                boxTL_x = int(boxBL_x - self.trackerParams["trackingBoxWidth"]/2 * self.trackerParams["sacleRatio"])
                boxTR_x = int(boxTL_x + self.trackerParams["trackingBoxWidth"] * self.trackerParams["sacleRatio"])
                boxT_y = self.trackerParams["bottomOffset"]
                boxB_y = self.imgHeight - self.trackerParams["topOffset"]
                # store the corner points
                self.rowTrackingBoxes.append(Polygon([(boxBR_x, boxB_y),
                                                        (boxBL_x, boxB_y), 
                                                        (boxTL_x, boxT_y),
                                                        (boxTR_x, boxT_y)]))

        else:
                        # window corner points are left and right of these
            boxBL_x = int(lineIntersection[0] - self.trackerParams["trackingBoxWidth"]/2)
            boxBR_x = int(boxBL_x + self.trackerParams["trackingBoxWidth"])
            boxTL_x = int(lineIntersection[1] - self.trackerParams["trackingBoxWidth"]/2 * self.trackerParams["sacleRatio"])
            boxTR_x = int(boxTL_x + self.trackerParams["trackingBoxWidth"] * self.trackerParams["sacleRatio"])
            boxT_y = self.trackerParams["bottomOffset"]
            boxB_y = self.imgHeight - self.trackerParams["topOffset"]
            # store the corner points
            self.rowTrackingBoxes[boxIdx] = Polygon([(boxBR_x, boxB_y),
                                                    (boxBL_x, boxB_y),
                                                    (boxTL_x, boxT_y),
                                                    (boxTR_x, boxT_y)])

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
    def trackError(self):
        cross_track_error = 0
        heading_error = 0

        ## plot robot heading and position
        cv2.arrowedLine(self.primaryImg, (int(self.imgWidth/2), self.imgHeight), (int(self.imgWidth/2), int(self.imgHeight/2)), (0, 0, 255), 2)
        # calculate cross-track error and heading error

        # put text on the image stating the cross-track error and heading error
        cv2.putText(self.primaryImg, f"Cross-Track Error: {cross_track_error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(self.primaryImg, f"Heading Error: {heading_error}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # display the image
        cv2.imshow("Primary Image", self.primaryImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return cross_track_error, heading_error
    def trackCropLane(self):
        # track crop lane using tracking boxes
        # return tracking boxes
        P, ang = None, None
        # if the feature extractor is initalized
        if self.cropLaneFound:
            # get the lines at windows defined through previous found lines
            lines, linesROIs, meanLinesInWindows = self.findLinesInImage()
            # if 'all' lines are found by 'self.getLinesInImage'
            if len(lines) >= len(self.trackingBoxLoc):
                # the line parameters are the new tracked lines (for the next step)
                self.CropRows = lines
                # location is always the left side of the window
                self.trackingBoxLoc = meanLinesInWindows - self.trackerParams["trackingBoxWidth"]/2
                # average image intersections of all found lines
                avgOfLines = np.mean(np.c_[utils.lineIntersectImgUpDown(
                    self.CropRows[:, 1], self.CropRows[:, 0], self.imgHeight)], axis=0)
                #  get AvgLine in image cords
                self.mainLine_up, self.mainLine_down = utils.getImgLineUpDown(
                    avgOfLines, self.imgHeight)
                # compute all intersections between the image and each line
                allLineIntersect = np.c_[utils.lineIntersectWin(self.CropRows[:, 1],
                                                          self.CropRows[:, 0],
                                                          self.imgHeight,
                                                          self.trackerParams["topOffset"],
                                                          self.trackerParams["bottomOffset"])]
                # store start and end points of lines - mainly for plotting
                self.allLineStart = np.c_[
                    allLineIntersect[:, 0], self.imgHeight - self.trackerParams["bottomOffset"] * np.ones((len(self.CropRows), 1))]
                self.allLineEnd = np.c_[
                    allLineIntersect[:, 1], self.trackerParams["topOffset"] * np.ones((len(self.CropRows), 1))]
                # main features
                self.P = self.cameraToImage([avgOfLines[1], self.imgHeight])
                self.ang = utils.computeTheta(self.mainLine_up, self.mainLine_down)
                self.cropLaneFound = True

            else:
                print("#[ERR] Lost at least one line")
                self.cropLaneFound = False
        else:
            print('Running rest()..')
        
        if self.pointsInBottom == 0 and self.pointsInTop == 0:
            self.cropLaneFound = False

        # if self.pointsInBottom == 0 and mode in [2, 5]:
        #     self.cropRowEnd = True
        # elif self.pointsInTop == 0 and mode in [1, 4]:
        #     self.cropRowEnd = True
        # else:
        #     self.cropRowEnd = False

        return self.cropLaneFound, P, ang
        

    def movingstd(self, signal, window=5):
        # calculate moving std deviation
        # return moving std deviation
        data = signal.copy()
        stdVec = np.zeros(((len(data) - window),1))
        for i in range(0, len(stdVec)):
            windowData = data[i:i+window]
            stdVec[i] = np.std(windowData)
        stdVec = stdVec/max(stdVec)
        stdVec = stdVec.reshape(len(stdVec),)
        return stdVec

    def findPeaksTroughs(self, stdvec, prominence=0.5, height=None):
        # find peaks and troughs in signal
        # return peaks and troughs
        stddev = stdvec.copy()
        troughs, _ = find_peaks(-stddev, height=height)
        peaks, _ = find_peaks(stddev, prominence=0.5, height=height)
        return peaks, troughs

    def cameraToImage(self, P):
        """function to transform the feature point from camera to image frame
        Args:
            P (_type_): point in camera Fr
        Returns:
            _type_: point in image Fr
            """
        P[0] = P[0] - self.imgWidth/2
        P[1] = P[1] - self.imgHeight/2
        return P

    def drawGraphics(self):
        """function to draw the lines and the windows onto the image (self.primaryRGBImg)
        """
        #print("DEBUG: drawGraphics")
        if self.primaryImg is not None:
            self.graphicsImg = self.primaryImg.copy()
            # main line
            # cv2.line(self.graphicsImg, (int(self.mainLine_up[0]), int(self.mainLine_up[1])), (int(
            #     self.mainLine_down[0]), int(self.mainLine_down[1])), (255, 0, 0), thickness=3)
            # contoures
            cv2.drawContours(self.graphicsImg,
                            self.plantObjects, -1, (10, 50, 150), 3)
            # for i in range(0, len(self.allLineStart)):
            #     # helper lines
            #     cv2.line(self.graphicsImg, (int(self.allLineStart[i, 0]), int(self.allLineStart[i, 1])), (int(
            #         self.allLineEnd[i, 0]), int(self.allLineEnd[i, 1])), (0, 255, 0), thickness=1)

            for i in range(self.numOfCropRows):
                int_coords = lambda x: np.array(x).round().astype(np.int32)
                exterior = [int_coords(self.rowTrackingBoxes[i].exterior.coords)]
                cv2.polylines(self.graphicsImg, exterior, True, (0, 255, 255))

            for i in range(len(self.plantCenters[0])):
                # draw point on countur centers
                x = int(self.plantCenters[0, i])
                y = int(self.plantCenters[1, i])
                self.graphicsImg = cv2.circle(
                    self.graphicsImg, (x, y), 3, (255, 0, 255), 5)
            cv2.imshow("Final draw image no lines found", self.graphicsImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: No primary image found")


if __name__ == "__main__":
    LaneDetector = CropLaneDetector()
    if LaneDetector.findCropLane():
        print("Crop Lane Found")
        #LaneDetector.trackCropLane()
    LaneDetector.trackError() # ideally to be called in the if condition but for visualization purposes called here