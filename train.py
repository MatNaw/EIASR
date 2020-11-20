import os
import random

from csv import reader

from usefulFunctions import Iou, getBox

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from math import sqrt
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

trainPath = './train_images'
valPath = './val_images'
labelPath = './labels.csv'
featureWeightsPath = './xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


trainList = os.listdir(trainPath)
valList = os.listdir(valPath)

with open(labelPath, 'r') as read_obj:
    csv_reader = reader(read_obj)
    labels = list(csv_reader)

#label = [name, typ, xmin, xmax, ymin, ymax]
#labels is a matrix: n - rows, 6-columns
#delete header
labels = labels[1:] 
#every second row is empty... don't ask why... so I delete it
del labels[::2]

#Scaling to uniform image size
uniformImgSize = (400,400)

#BOXES
boxSizes = [int(uniformImgSize[0]/5*3), int(uniformImgSize[0]/5*2), int(uniformImgSize[0]/5)]
boxScales= [1,1/(sqrt(2)), sqrt(2)] #Scales of boxes: 1:1, 1:2, 2:1

#ANCHORS
anchorStepX = 5 #pixels
anchorStepY = 5 #pixels
firstAnchorX = 5
firstAnchorY = 5



#get an image from training pool
sampleName = random.choice(trainList)
# sampleName = 'drones-inspire-phantom-mavic-on-260nw-1139013731.jpg' # GOOD EXAMPLE UNCOMMENT IF YOU WANT TO UNDERSTAND
sampleImage = cv2.imread(trainPath + '/' + sampleName)
sampleImageResized = cv2.resize(sampleImage, uniformImgSize, interpolation=cv2.INTER_AREA)
(yS,xS,_) = sampleImage.shape           #S - Sample
(yR,xR,_) = sampleImageResized.shape    #R - Resized
xRatio = xS/xR
yRatio = yS/yR

# cv2.imshow('1',sampleImage)
# cv2.imshow('2',sampleImageResized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Build anchorBoxes for an image and create positive and negative ones
positiveBoxes=[]
negativeBoxes=[]
realBoxesResized = []

#find real boxes and resize them
for label in labels:
    if label[0] == sampleName:
        realBoxesResized.append([round(int(label[2])/xRatio), round(int(label[3])/xRatio), round(int(label[4])/yRatio), round(int(label[5])/yRatio)])



anchorsAlongX = round((xR - firstAnchorX)/anchorStepX) +1 #On scaled image
anchorsAlongY = round((yR - firstAnchorY)/anchorStepY) +1 #On scaled image

#IoU needed to accept that a box contains a drone - should not be too high
positiveBoxThreshold = 0.4

for i in range(anchorsAlongX):
    for j in range(anchorsAlongY):
        for boxSize in boxSizes:
            for boxScale in boxScales:
                currIoUs=[] #Important for multiple drones in one image
                width = round(boxSize * boxScale)
                height = round(boxSize / boxScale)
                anchorX = firstAnchorX + i * anchorStepX
                anchorY = firstAnchorY + j * anchorStepY
                (xmin, xmax, ymin, ymax) = getBox(anchorX, anchorY, width, height, (yR,xR,3))
                for realBoxR in realBoxesResized:
                    currIou = Iou([xmin, xmax, ymin, ymax], realBoxR)
                    currIoUs.append(currIou)
                if max(currIoUs) >= positiveBoxThreshold:
                    positiveBoxes.append([xmin, xmax, ymin, ymax])
                else:
                    negativeBoxes.append([xmin, xmax, ymin, ymax])

#since we have boxes let's pick one positive and one negative
samplePositiveBox = random.choice(positiveBoxes)
sampleNegativeBox = random.choice(negativeBoxes)

#take part of image for those boxes
positiveImagePart = sampleImageResized[samplePositiveBox[2]:samplePositiveBox[3], samplePositiveBox[0]:samplePositiveBox[1]]
negativeImagePart = sampleImageResized[sampleNegativeBox[2]:sampleNegativeBox[3], sampleNegativeBox[0]:sampleNegativeBox[1]]



### NOT NEEDED PART FOR WORKING BEGIN ###
# TO SEE!!! WHAT IS HAPPENING TILL NOW
fig,ax = plt.subplots(3)
ax[0].imshow(sampleImageResized)
rect = patches.Rectangle((samplePositiveBox[0], samplePositiveBox[2]),samplePositiveBox[1]-samplePositiveBox[0],samplePositiveBox[3]-samplePositiveBox[2],linewidth=1,edgecolor='b',facecolor='none')
ax[0].add_patch(rect)
rect = patches.Rectangle((sampleNegativeBox[0], sampleNegativeBox[2]),sampleNegativeBox[1]-sampleNegativeBox[0],sampleNegativeBox[3]-sampleNegativeBox[2],linewidth=1,edgecolor='r',facecolor='none')
ax[0].add_patch(rect)
rect = patches.Rectangle((realBoxesResized[0][0], realBoxesResized[0][2]),realBoxesResized[0][1]-realBoxesResized[0][0],realBoxesResized[0][3]-realBoxesResized[0][2],linewidth=1,edgecolor='g',facecolor='none')
ax[0].add_patch(rect)
ax[1].imshow(positiveImagePart)
ax[2].imshow(negativeImagePart)
ax[0].set_title('Original')
ax[1].set_title('Positive')
ax[2].set_title('Negative')
plt.show()
cv2.waitKey(0)
### NOT NEEDED PART FOR WORKING END ###



# #make dataset
# positiveImagePart = cv2.resize(positiveImagePart, (boxSizes[2],boxSizes[2]), interpolation=cv2.INTER_AREA)
# negativeImagePart = cv2.resize(negativeImagePart, (boxSizes[2],boxSizes[2]), interpolation=cv2.INTER_AREA)
# positiveImagePart = np.expand_dims(positiveImagePart, axis=0)
# negativeImagePart = np.expand_dims(negativeImagePart, axis=0)
# dataset = ([tf.cast(positiveImagePart, tf.float32), tf.cast(negativeImagePart, tf.float32)], [1, 0])



# ### MODEL BUILDING BEGIN ###
# # FEATURE EXTRACTION LAYERS - TRANSFER LEARNING
# input = Input(shape=(boxSizes[2], boxSizes[2],3)) #input shape


# base_model = Xception(
#     include_top=False, #no dense layers in the end to classify so i can make my own
#     weights=featureWeightsPath,
#     #input_tensor = input
#     input_shape = (boxSizes[2], boxSizes[2],3)
# )
# base_model.trainable=False

# #OUR SEGMENT OF NETWORK - TRAINABLE
# x = base_model(input, training=False)
# x = GlobalAveragePooling2D()(x)
# output = Dense(1)(x)
# model = Model(input, output)

# #loss function
# loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
# #optimizer
# optimizer = keras.optimizers.Adam()
# ### MODEL BUILDING END ###


# for inputs, targets in dataset:
#     with tf.GradientTape() as tape:
#         predictions = model(inputs)
#         loss_value = loss_fn(targets, predictions)
#     gradients = tape.gradient(loss_value, model.trainable_weights)
#     optimizer.apply_gradients(zip(gradients, model.trainable_weights))

# # Iterate over the batches of a dataset.
# for inputs, targets in dataset:
#     # Open a GradientTape.
#     with tf.GradientTape() as tape:
#         # Forward pass.
#         predictions = model(inputs)
#         # Compute the loss value for this batch.
#         loss_value = loss_fn(targets, predictions)

#     # Get gradients of loss wrt the *trainable* weights.
#     gradients = tape.gradient(loss_value, model.trainable_weights)
#     # Update the weights of the model.
#     optimizer.apply_gradients(zip(gradients, model.trainable_weights))
print('GOTOWE')