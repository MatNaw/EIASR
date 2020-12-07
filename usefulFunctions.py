import cv2
import random

#IoU intersection over union
#box = [xmin, xmax, ymin, ymax]
def Iou(box1, box2):
    box1Area = (box1[1] - box1[0])*(box1[3]-box1[2])
    box2Area = (box2[1] - box2[0])*(box2[3]-box2[2])

    xInterLeft = max(box1[0],box2[0])
    yInterLeft = max(box1[2],box2[2])
    xInterRight = min(box1[1],box2[1])
    yInterRight = min(box1[3],box2[3])
    intersectionArea = max(0,xInterRight-xInterLeft)*max(0,yInterRight-yInterLeft)
    unionArea = box1Area+box2Area-intersectionArea
    return intersectionArea / unionArea

#Przycina box-y na skraju obrazka
def getBox(anchorX, anchorY, boxWidth, boxHeight, imgShape):
    (imgY, imgX, _) = imgShape
    #cutting on egdes
    xmin = max(0, anchorX - round(boxWidth/2))
    xmax = min(imgX, anchorX + round(boxWidth/2))
    ymin = max(0, anchorY - round(boxHeight/2))
    ymax = min(imgY, anchorY + round(boxHeight/2))
    return (xmin, xmax, ymin, ymax)


#return positive and negative boxes, marking respective to positiveBoxThreshold
def markBoxes(xR, yR, boxSizes, boxScales, realBoxesResized, positiveBoxThreshold, firstAnchorX = 5, firstAnchorY = 5, StepX = 5, StepY = 5):

    anchorsAlongX = round((xR - firstAnchorX)/StepX) +1 #On scaled image
    anchorsAlongY = round((yR - firstAnchorY)/StepY) +1 #On scaled image

    positiveBoxes=[]
    negativeBoxes=[]

    for i in range(anchorsAlongX):
        for j in range(anchorsAlongY):
            for boxSize in boxSizes:
                for boxScale in boxScales:
                    currIoUs=[] #Important for multiple drones in one image
                    width = round(boxSize * boxScale)
                    height = round(boxSize / boxScale)
                    anchorX = firstAnchorX + i * StepX
                    anchorY = firstAnchorY + j * StepY
                    (xmin, xmax, ymin, ymax) = getBox(anchorX, anchorY, width, height, (yR,xR,3))
                    for realBoxR in realBoxesResized:
                        currIou = Iou([xmin, xmax, ymin, ymax], realBoxR)
                        currIoUs.append(currIou)
                    if max(currIoUs) >= positiveBoxThreshold:
                        positiveBoxes.append([xmin, xmax, ymin, ymax])
                    else:
                        negativeBoxes.append([xmin, xmax, ymin, ymax])
    return positiveBoxes, negativeBoxes


#JESZCZE NIE DZIALA POPRAWNIE
def createBatch(batchSize, trainList, trainPath, labels, boxSizes, boxScales, uniformImgSize = (400,400), positiveBoxThreshold = 0.4):

    batchImgs = []
    batchLabels = []
    xR = uniformImgSize[0]
    yR = uniformImgSize[1]

    while len(batchImgs) < batchSize:
        sampleName = random.choice(trainList)
        sampleImage = cv2.imread(trainPath + '/' + sampleName)
        (yS,xS,_) = sampleImage.shape
        sampleImageResized = cv2.resize(sampleImage, uniformImgSize, interpolation=cv2.INTER_AREA)
        xRatio = xS/xR
        yRatio = yS/yR

        #getting realBoxes from image' labels
        realBoxesResized = []
        for label in labels:
            if label[0] == sampleName:
                realBoxesResized.append([round(int(label[2])/xRatio), round(int(label[3])/xRatio), round(int(label[4])/yRatio), round(int(label[5])/yRatio)])

        #marked boxes
        positiveBoxes, negativeBoxes = markBoxes(xR, yR, boxSizes, boxScales, realBoxesResized, positiveBoxThreshold)
        print("negative length:")
        print(len(negativeBoxes))
        print("positive length:")
        print(len(positiveBoxes))
        print("\n\nACK0\n\n")
        print(len(batchImgs))
        print("\n\n")
        print(len(batchLabels))
        #creating a batch
        if 2*len(positiveBoxes) <= (batchSize - len(batchImgs)):
            print("\n\nACK1\n\n")
            for box in positiveBoxes:
                batchImgs.append(sampleImageResized[box[2]:box[3], box[0]:box[1],0:3])
                batchLabels.append([1])
            negativeBoxes = random.sample(negativeBoxes, len(positiveBoxes)) #same amount of negative and positive from an image
            for box in negativeBoxes:
                batchImgs.append(sampleImageResized[box[2]:box[3], box[0]:box[1],0:3])
                batchLabels.append([0])
        else:
            print("\n\nACK2\n\n")
            positiveBoxes = random.sample(positiveBoxes, int((batchSize - len(batchImgs))/2))
            for box in positiveBoxes:
                batchImgs.append(sampleImageResized[box[2]:box[3], box[0]:box[1],0:3])
                batchLabels.append([1])
            negativeBoxes = random.sample(negativeBoxes, int((batchSize - len(batchImgs))/2)) #same amount of negative and positive from an image
            for box in negativeBoxes:
                batchImgs.append(sampleImageResized[box[2]:box[3], box[0]:box[1],0:3])
                batchLabels.append([0])

    return batchImgs, batchLabels