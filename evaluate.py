import os
import random
import numpy as np
from csv import reader

from useful_functions import cut_on_edges, NMS
from constants import TO_PREDICT_PATH, PREDICTED_PATH, UNIFORM_IMG_SIZE, FIRST_ANCHOR_X, FIRST_ANCHOR_Y, ANCHOR_STEP_X, ANCHOR_STEP_Y, BOX_SIZES, BOX_SCALES, KERAS_IMG_SIZE
from train import build_model
import cv2




def predict_images():
    pred_list = os.listdir(TO_PREDICT_PATH)

    (_, model) = build_model()
    model.load_weights('./model.h5')


    (y_resized, x_resized) = UNIFORM_IMG_SIZE

    for image in pred_list:
        sample_image1 = cv2.imread(TO_PREDICT_PATH + '/' + image)
        sample_image = sample_image1.astype(np.float32)/255.0
        (y_sample_shape, x_sample_shape, _) = sample_image.shape
        sample_image_resized = cv2.resize(sample_image, UNIFORM_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        x_ratio = x_sample_shape / x_resized
        y_ratio = y_sample_shape / y_resized

        anchors_along_x = int((x_resized - FIRST_ANCHOR_X) / ANCHOR_STEP_X) + 1  # On scaled image
        anchors_along_y = int((y_resized - FIRST_ANCHOR_Y) / ANCHOR_STEP_Y) + 1  

        Drones = []
        Drones_marks = []

        boxes = []
        predictions = []

        iterator = 0
        keras_input = []
        iterator_max = 100

        bol = 1

        k=0
        remaining = anchors_along_x*anchors_along_y*len(BOX_SCALES)*len(BOX_SIZES)
        print("Remaining: %d" % remaining)
        for i in range(anchors_along_x):
            for j in range(anchors_along_y):
                for boxSize in BOX_SIZES:
                    for boxScale in BOX_SCALES:
                        width = round(boxSize * boxScale)
                        height = round(boxSize / boxScale)
                        anchor_x = FIRST_ANCHOR_X + i * ANCHOR_STEP_X
                        anchor_y = FIRST_ANCHOR_Y + j * ANCHOR_STEP_Y
                        box = cut_on_edges(UNIFORM_IMG_SIZE, [anchor_x - width/2, anchor_x + width/2, anchor_y - height/2, anchor_y + height/2])
                        boxes.append(box)

                        keras_input.append(cv2.resize(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3], KERAS_IMG_SIZE, interpolation=cv2.INTER_CUBIC))
                        iterator = iterator + 1

                        if remaining >= iterator_max:
                            if iterator == iterator_max:
                                y_pred = model.predict_on_batch(np.array(keras_input))
                                for pred in y_pred:
                                    if np.argmax(pred) == 1:
                                        if pred[1] >= 0.9:
                                            Drones.append(boxes[k])
                                            Drones_marks.append(pred[1])
                                    k = k + 1
                                keras_input = []
                                remaining = remaining - iterator
                                iterator = 0
                                print("REMAINING: %d" % remaining)
                        else:
                            if iterator == remaining:
                                y_pred = model.predict_on_batch(np.array(keras_input))
                                for pred in y_pred:
                                    if np.argmax(pred) == 1:
                                        if pred[1] >= 0.9:
                                            Drones.append(boxes[k])
                                            Drones_marks.append(pred[1])
                                    k = k + 1                
                                keras_input = []
                                remaining = remaining - iterator
                                iterator = 0
                                print("REMAINING: %d" % remaining)
                        

                        
        # argmax = np.argmax(predictions)
        # liczba = -100
        # l=0
        Drones = NMS(Drones, Drones_marks, IoU_threshold=0.1)

        for drone in Drones:
            drone = [int(drone[0]*x_ratio), int(drone[1]*x_ratio), int(drone[2]*y_ratio), int(drone[3]*y_ratio)]
            cv2.rectangle(sample_image1, (drone[0], drone[2]), (drone[1], drone[3]), (255,0,0), 2)
        # our_box = boxes[argmax]

        # our_box = [int(our_box[0]*x_ratio), int(our_box[1]*x_ratio), int(our_box[2]*y_ratio), int(our_box[3]*y_ratio)]

        # cv2.rectangle(sample_image1, (our_box[0], our_box[2]), (our_box[1], our_box[3]), (255,0,0), 2)
        cv2.imwrite(PREDICTED_PATH + '/' + image, sample_image1)
        print(len(Drones))
        print("NEXT DONE")


    print("Predicted!")