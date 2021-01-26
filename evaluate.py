import os
import numpy as np
import pandas as pd

from useful_functions import cut_on_edges, NMS
from constants import TO_PREDICT_PATH, PREDICTED_PATH, UNIFORM_IMG_SIZE, FIRST_ANCHOR_X, FIRST_ANCHOR_Y, \
    ANCHOR_STEP_X, ANCHOR_STEP_Y, BOX_SIZES, BOX_SCALES, KERAS_IMG_SIZE, CLASS_MODEL_PATH, REGRESSOR_MODEL_PATH, \
    LABEL_PATH
from train import build_models
import cv2


def get_label_anchor_box(image_name):
    padnads_csv_file = pd.read_csv(LABEL_PATH)
    return padnads_csv_file.loc[padnads_csv_file['image_names'] == image_name][['xmin', 'xmax', 'ymin', 'ymax']].to_numpy()[0]


def predict_images():
    pred_list = os.listdir(TO_PREDICT_PATH)

    _, classifier, regressor = build_models()
    classifier.load_weights(CLASS_MODEL_PATH)
    regressor.load_weights(REGRESSOR_MODEL_PATH)
    (y_resized, x_resized) = UNIFORM_IMG_SIZE

    for image in pred_list:
        sample_image1 = cv2.imread(TO_PREDICT_PATH + '/' + image)
        sample_image = sample_image1.astype(np.float32) / 255.0

        (y_sample_shape, x_sample_shape, _) = sample_image.shape
        sample_image_resized = cv2.resize(sample_image, UNIFORM_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        x_ratio = x_sample_shape / x_resized
        y_ratio = y_sample_shape / y_resized

        anchors_along_x = int((x_resized - FIRST_ANCHOR_X) / ANCHOR_STEP_X) + 1  # On scaled image
        anchors_along_y = int((y_resized - FIRST_ANCHOR_Y) / ANCHOR_STEP_Y) + 1  

        Drones = []
        Drones_marks = []

        boxes = []
        iterator = 0
        keras_input = []
        iterator_max = 100

        remaining_anchors = anchors_along_x * anchors_along_y * len(BOX_SCALES) * len(BOX_SIZES)
        print("Initial remaining anchors to iterate: ", remaining_anchors)

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

                        keras_input.append(cv2.resize(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3],
                                                      KERAS_IMG_SIZE,
                                                      interpolation=cv2.INTER_CUBIC))
                        iterator += 1

                        if iterator == min(remaining_anchors, iterator_max):
                            k = 0
                            classifier_pred = classifier.predict_on_batch(np.array(keras_input))
                            regressor_pred = regressor.predict_on_batch(np.array(keras_input))

                            for pred in classifier_pred:
                                if np.argmax(pred) == 1:
                                    if pred[1] >= 0.9:
                                        [box_x_min, _, box_y_min, _] = boxes[k]
                                        [reg_box_x_min, reg_box_x_max, reg_box_y_min, reg_box_y_max] = \
                                            regressor_pred[k] * int(UNIFORM_IMG_SIZE[0] / KERAS_IMG_SIZE[0])

                                        reg_box_x_min += box_x_min
                                        reg_box_x_max += box_x_min
                                        reg_box_y_min += box_y_min
                                        reg_box_y_max += box_y_min

                                        Drones.append([reg_box_x_min, reg_box_x_max, reg_box_y_min, reg_box_y_max])
                                        Drones_marks.append(pred[1])
                                k = k + 1
                            keras_input = []
                            boxes = []
                            remaining_anchors -= iterator
                            iterator = 0
                            print("Remaining anchors to iterate: ", remaining_anchors)

        Drones = NMS(Drones, Drones_marks, IoU_threshold=0.1)

        # draw the predicted rectangles
        for drone in Drones:
            drone = [int(drone[0] * x_ratio), int(drone[1] * x_ratio), int(drone[2] * y_ratio), int(drone[3] * y_ratio)]
            cv2.rectangle(sample_image1, (drone[0], drone[2]), (drone[1], drone[3]), (255, 0, 0), 2)
            labeled_anchor_box = get_label_anchor_box(image)
            cv2.rectangle(sample_image1, (labeled_anchor_box[0], labeled_anchor_box[2]), (labeled_anchor_box[1], labeled_anchor_box[3]), (0, 255, 0), 2)

        cv2.imwrite(PREDICTED_PATH + '/' + image, sample_image1)
        print("Length of Drones: " + str(len(Drones)))
        print("Next image...")

    print("All images predicted!")
