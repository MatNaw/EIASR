import os
import random

from csv import reader

from usefulFunctions import Iou, getBox, markBoxes, createBatch

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
# anchorStepX = 5 #pixels
# anchorStepY = 5 #pixels
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

#Build anchorBoxes for an image
realBoxesResized = []

#find real boxes and resize them
for label in labels:
    if label[0] == sampleName:
        realBoxesResized.append([round(int(label[2])/xRatio), round(int(label[3])/xRatio), round(int(label[4])/yRatio), round(int(label[5])/yRatio)])

#IoU needed to accept that a box contains a drone - should not be too high
positiveBoxThreshold = 0.4

positiveBoxes, negativeBoxes = markBoxes(xR, yR, boxSizes, boxScales, realBoxesResized, positiveBoxThreshold)

#since we have boxes let's pick one positive and one negative
samplePositiveBox = random.choice(positiveBoxes)
sampleNegativeBox = random.choice(negativeBoxes)

#take part of image for those boxes
positiveImagePart = sampleImageResized[samplePositiveBox[2]:samplePositiveBox[3], samplePositiveBox[0]:samplePositiveBox[1],0:3]
negativeImagePart = sampleImageResized[sampleNegativeBox[2]:sampleNegativeBox[3], sampleNegativeBox[0]:sampleNegativeBox[1],0:3]



# ### NOT NEEDED PART FOR WORKING BEGIN ###
# # TO SEE!!! WHAT IS HAPPENING TILL NOW
# fig,ax = plt.subplots(3)
# ax[0].imshow(sampleImageResized)
# rect = patches.Rectangle((samplePositiveBox[0], samplePositiveBox[2]),samplePositiveBox[1]-samplePositiveBox[0],samplePositiveBox[3]-samplePositiveBox[2],linewidth=1,edgecolor='b',facecolor='none')
# ax[0].add_patch(rect)
# rect = patches.Rectangle((sampleNegativeBox[0], sampleNegativeBox[2]),sampleNegativeBox[1]-sampleNegativeBox[0],sampleNegativeBox[3]-sampleNegativeBox[2],linewidth=1,edgecolor='r',facecolor='none')
# ax[0].add_patch(rect)
# rect = patches.Rectangle((realBoxesResized[0][0], realBoxesResized[0][2]),realBoxesResized[0][1]-realBoxesResized[0][0],realBoxesResized[0][3]-realBoxesResized[0][2],linewidth=1,edgecolor='g',facecolor='none')
# ax[0].add_patch(rect)
# ax[1].imshow(positiveImagePart)
# ax[2].imshow(negativeImagePart)
# ax[0].set_title('Original')
# ax[1].set_title('Positive')
# ax[2].set_title('Negative')
# plt.show()
# cv2.waitKey(0)
# ### NOT NEEDED PART FOR WORKING END ###


input_shape = int(uniformImgSize[0]/5)
#make dataset
positiveImagePart = cv2.resize(positiveImagePart, (input_shape,input_shape), interpolation=cv2.INTER_AREA)
positiveImagePart = np.array(positiveImagePart)
positiveImagePart.astype('float32')
positiveImagePart = positiveImagePart/255

negativeImagePart = cv2.resize(negativeImagePart, (input_shape,input_shape), interpolation=cv2.INTER_AREA)
nevativeImagePart = np.array(negativeImagePart)
negativeImagePart.astype('float32')
negativeImagePart = negativeImagePart/255
# positiveImagePart = np.expand_dims(positiveImagePart, axis=0)
# negativeImagePart = np.expand_dims(negativeImagePart, axis=0)
datasetImg = []
datasetLabels = []
datasetImg.append(positiveImagePart)
datasetLabels.append([1])
datasetImg.append(negativeImagePart)
datasetLabels.append([0])

###
# model=tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(input_shape,input_shape, 3)),
#             tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
#             tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(2)
#         ])

# model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x=np.array(datasetImg, np.float32), y=np.array(datasetLabels, np.float32), epochs=5)

# print(model.predict(x=np.array(datasetImg, np.float32)))
###



# ### MODEL BUILDING BEGIN ###
# # FEATURE EXTRACTION LAYERS - TRANSFER LEARNING

input = Input(shape=(boxSizes[2], boxSizes[2],3)) #input shape


base_model = Xception(
    include_top=False, #no dense layers in the end to classify so i can make my own
    weights=featureWeightsPath,
    input_shape = (input_shape, input_shape,3)
)
base_model.trainable=False

#OUR SEGMENT OF NETWORK - TRAINABLE
x = base_model(input, training=False)
x = GlobalAveragePooling2D()(x)
output = Dense(1)(x)
model = Model(input, output)

# #loss function
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
# #optimizer
optimizer = keras.optimizers.Adam()
# ### MODEL BUILDING END ###


with tf.GradientTape() as tape:
    predictions = model(np.array(datasetImg, np.float32))
    loss_value = loss_fn(np.array(datasetLabels, np.float32), predictions)
gradients = tape.gradient(loss_value, model.trainable_weights)
optimizer.apply_gradients(zip(gradients, model.trainable_weights))

print(loss_value)
print(model.predict(np.array(datasetImg, np.float32)))

#MASS TRAINING
BATCH_SIZE = 2
NUM_EPOCHS = 100


# for i in range(NUM_EPOCHS):
#     #Get a batch of data
#     PositiveSamples = []
#     NegativeSamples = []
#     while len(PositiveSamples) < BATCH_SIZE/2:
#         sampleName = random.choice(trainList)


# model.compile(optimizer=optimizer, loss=loss_fn)
# test_loss, test_accuracy = model.evaluate(np.array(datasetImg, np.float32),  np.array(datasetLabels, np.float32), verbose=2)
# print(test_loss)
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

#JESZCZE NIE DZIALA
# batchImgs, batchLabels = createBatch(BATCH_SIZE, trainList, trainPath, labels, boxSizes, boxScales)
# print(batchLabels)
print('GOTOWE')