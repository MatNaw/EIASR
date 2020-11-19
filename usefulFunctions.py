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