import os
import cv2
import random
import numpy as np

from constants import TRAIN_PATH, UNIFORM_IMG_SIZE, EPOCH_LENGTH, BATCH_SIZE, BOX_SIZES, BOX_SCALES, KERAS_IMG_SIZE, \
    FIRST_ANCHOR_X, ANCHOR_STEP_X, ANCHOR_STEP_Y, FIRST_ANCHOR_Y, VAL_PATH


def calculate_IoU(box1, box2):
    """
    Calculates Intersection over Union (IoU) metric. Box is defined as a rectangle:
    box = [xmin, xmax, ymin, ymax]
    :param box1: First rectangle box considered in evaluation of IoU
    :param box2: Second rectangle box considered in evaluation of IoU
    :return: IoU metric value of boxes box1 and box2
    """
    box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
    box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])

    x_inter_left = max(box1[0], box2[0])
    y_inter_left = max(box1[2], box2[2])
    x_inter_right = min(box1[1], box2[1])
    y_inter_right = min(box1[3], box2[3])

    intersection_area = max(0, x_inter_right - x_inter_left) * max(0, y_inter_right - y_inter_left)
    union_area = box1_area + box2_area - intersection_area

    try:
        return intersection_area / union_area
    except ZeroDivisionError:
        print("IoU calculation error - union area is equal to 0!")
        return 0


def get_box(anchor_x, anchor_y, box_width, box_height, img_shape):
    """
    Cuts the given box on edges, according to the given image shape and anchor point.
    :param anchor_x: X coordinate of anchor point
    :param anchor_y: Y coordinate of anchor point
    :param box_width: Width of the box
    :param box_height: Height of the box
    :param img_shape: Image shape
    :return: Rectangle box (box = [xmin, xmax, ymin, ymax])
    """
    (img_y, img_x, _) = img_shape
    half_box_width = round(box_width / 2)
    half_box_height = round(box_height / 2)

    # cutting on edges
    x_min = max(0, anchor_x - half_box_width)
    x_max = min(img_x, anchor_x + half_box_width)
    y_min = max(0, anchor_y - half_box_height)
    y_max = min(img_y, anchor_y + half_box_height)

    return x_min, x_max, y_min, y_max


def mark_boxes(x_resized, y_resized, real_boxes_resized, positive_box_threshold, negative_box_threshold):
    """
    Returns positive and negative boxes, marking them respectively to positive and negative box threshold.
    :param x_resized: Resized shape of original image in X axis
    :param y_resized: Resized shape of original image in Y axis
    :param real_boxes_resized: List of real boxes (retrieved from labels.csv)
    :param positive_box_threshold: Positive threshold to assign the box to the list of positive samples
    :param negative_box_threshold: Negative threshold to assign the box to the list of negative samples
    :return: Two lists of rectangle boxes (box = [xmin, xmax, ymin, ymax])
    """
    anchors_along_x = round((x_resized - FIRST_ANCHOR_X) / ANCHOR_STEP_X) + 1  # On scaled image
    anchors_along_y = round((y_resized - FIRST_ANCHOR_Y) / ANCHOR_STEP_Y) + 1  # On scaled image

    positive_boxes = []
    negative_boxes = []

    for i in range(anchors_along_x):
        for j in range(anchors_along_y):
            for boxSize in BOX_SIZES:
                for boxScale in BOX_SCALES:
                    IoU_values = []  # important for detecting multiple drones in one image
                    width = round(boxSize * boxScale)
                    height = round(boxSize / boxScale)
                    anchor_x = FIRST_ANCHOR_X + i * ANCHOR_STEP_X
                    anchor_y = FIRST_ANCHOR_Y + j * ANCHOR_STEP_Y
                    (x_min, x_max, y_min, y_max) = get_box(anchor_x, anchor_y, width, height, (y_resized, x_resized, 3))
                    box_w = x_max - x_min
                    box_h = y_max - y_min

                    for realBoxR in real_boxes_resized:
                        current_IoU = calculate_IoU([x_min, x_max, y_min, y_max], realBoxR)
                        IoU_values.append(current_IoU)

                    if max(IoU_values) >= positive_box_threshold:
                        [real_x_min, real_x_max, real_y_min, real_y_max] = real_boxes_resized[np.argmax(IoU_values)]
                        real_x_min = (real_x_min - x_min) / box_w
                        real_x_max = (real_x_max - x_min) / box_w
                        real_y_min = (real_y_min - y_min) / box_h
                        real_y_max = (real_y_max - y_min) / box_h

                        positive_boxes.append(([x_min, x_max, y_min, y_max], [real_x_min, real_x_max, real_y_min, real_y_max]))
                    elif max(IoU_values) <= negative_box_threshold:
                        negative_boxes.append([x_min, x_max, y_min, y_max])
    return positive_boxes, negative_boxes


def create_batch(labels, positive_box_threshold=0.7, negative_box_threshold=0.3):
    """
    Creates a single batch vector of images for batch training of the models.
    :param labels: Labels retrieved from labels.csv
    :param positive_box_threshold: Positive threshold to assign the box to the list of positive samples
    :param negative_box_threshold: Negative threshold to assign the box to the list of negative samples
    :return: Batch vector of images for classifier and regressor, list of labels and list of generated boxes
    """
    train_list = os.listdir(TRAIN_PATH)

    batch_classifier_images = []
    batch_regressor_images = []
    batch_labels = []
    batch_boxes = []
    x_resized = UNIFORM_IMG_SIZE[0]
    y_resized = UNIFORM_IMG_SIZE[1]

    while len(batch_classifier_images) < BATCH_SIZE:
        sample_name = random.choice(train_list)
        sample_image = cv2.imread(TRAIN_PATH + '/' + sample_name).astype(np.float32) / 255.0
        (y_sample_shape, x_sample_shape, _) = sample_image.shape
        sample_image_resized = cv2.resize(sample_image, UNIFORM_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        x_ratio = x_sample_shape / x_resized
        y_ratio = y_sample_shape / y_resized

        # getting real_boxes from image labels
        real_boxes_resized = []
        for label in labels:
            if label[0] == sample_name:
                real_boxes_resized.append([
                    round(int(label[2]) / x_ratio),
                    round(int(label[3]) / x_ratio),
                    round(int(label[4]) / y_ratio),
                    round(int(label[5]) / y_ratio)
                ])

        # marked boxes
        positive_boxes, negative_boxes = mark_boxes(x_resized, y_resized, real_boxes_resized,
                                                                          positive_box_threshold,
                                                                          negative_box_threshold)

        # creating a batch
        if 2 * len(positive_boxes) <= (BATCH_SIZE - len(batch_classifier_images)):
            for (box, real_box) in positive_boxes:
                resize_box_for_keras = cv2.resize(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3],
                                              KERAS_IMG_SIZE,
                                              interpolation=cv2.INTER_CUBIC)
                batch_classifier_images.append(resize_box_for_keras)
                batch_regressor_images.append(resize_box_for_keras)
                batch_labels.append([0.0, 1.0])
                batch_boxes.append(real_box)

            negative_boxes = random.sample(negative_boxes,
                                           len(positive_boxes))  # same amount of negative and positive from an image
            for box in negative_boxes:
                resize_box_for_keras = cv2.resize(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3],
                                              KERAS_IMG_SIZE,
                                              interpolation=cv2.INTER_CUBIC)
                batch_classifier_images.append(resize_box_for_keras)
                batch_labels.append([1.0, 0.0])
        else:
            positive_boxes = random.sample(positive_boxes, k=int((BATCH_SIZE - len(batch_classifier_images)) / 2))
            negative_boxes = random.sample(negative_boxes, k=int(
                (BATCH_SIZE - len(batch_classifier_images)) / 2))  # same amount of negative and positive from an image

            for (box, real_box) in positive_boxes:
                resize_box_for_keras = cv2.resize(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3],
                                                  KERAS_IMG_SIZE,
                                                  interpolation=cv2.INTER_CUBIC)
                batch_classifier_images.append(resize_box_for_keras)
                batch_regressor_images.append(resize_box_for_keras)
                batch_labels.append([0.0, 1.0])
                batch_boxes.append(real_box)

            for box in negative_boxes:
                resize_box_for_keras = cv2.resize(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3],
                                              KERAS_IMG_SIZE,
                                              interpolation=cv2.INTER_CUBIC)
                batch_classifier_images.append(resize_box_for_keras)
                batch_labels.append([1.0, 0.0])

    return batch_classifier_images, batch_regressor_images, batch_labels, batch_boxes


def create_batch_list(labels, positive_box_threshold=0.7, negative_box_threshold=0.3):
    """
    Creates a list of batches for batch training of the models.
    :param labels: Labels retrieved from labels.csv
    :param positive_box_threshold: Positive threshold to assign the box to the list of positive samples
    :param negative_box_threshold: Negative threshold to assign the box to the list of negative samples
    :return: Dataset presenting the list of batches for training
    """
    dataset = []

    for step in range(EPOCH_LENGTH):
        batch_classifier_images, batch_regressor_images, batch_labels, batch_boxes = \
            create_batch(labels, positive_box_threshold=positive_box_threshold, negative_box_threshold=negative_box_threshold)
        dataset.append((batch_classifier_images, batch_regressor_images, batch_labels, batch_boxes))

    return dataset


def create_test_data(labels, positive_box_threshold=0.7, negative_box_threshold=0.3, batch_size=100):
    """
    Creates a list of test data.
    :param labels: Labels retrieved from labels.csv
    :param positive_box_threshold: Positive threshold to assign the box to the list of positive samples
    :param negative_box_threshold: Negative threshold to assign the box to the list of negative samples
    :param batch_size: Size of test data batch
    :return: Test data vectors for classifier and regressor, vector of labels and vector of boxes
    """
    test_classifier_images = []
    test_regressor_images = []
    test_labels = []
    test_boxes = []

    test_list = os.listdir(VAL_PATH)

    x_resized = UNIFORM_IMG_SIZE[0]
    y_resized = UNIFORM_IMG_SIZE[1]

    while len(test_labels) < batch_size:
        sample_name = random.choice(test_list)
        sample_image = cv2.imread(VAL_PATH + '/' + sample_name).astype(np.float32) / 255.0
        (y_sample_shape, x_sample_shape, _) = sample_image.shape
        sample_image_resized = cv2.resize(sample_image, UNIFORM_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        x_ratio = x_sample_shape / x_resized
        y_ratio = y_sample_shape / y_resized

        real_boxes_resized = []
        for label in labels:
            if label[0] == sample_name:
                real_boxes_resized.append([
                    round(int(label[2]) / x_ratio),
                    round(int(label[3]) / x_ratio),
                    round(int(label[4]) / y_ratio),
                    round(int(label[5]) / y_ratio)
                ])

        positive_boxes, negative_boxes = mark_boxes(x_resized, y_resized, real_boxes_resized,
                                                                          positive_box_threshold,
                                                                          negative_box_threshold)

        if len(positive_boxes) == 0:
            continue

        (box, real_box) = random.choice(positive_boxes)
        resize_for_keras = cv2.resize(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3],
                                      KERAS_IMG_SIZE,
                                      interpolation=cv2.INTER_CUBIC)
        test_classifier_images.append(resize_for_keras)
        test_regressor_images.append(resize_for_keras)
        test_labels.append([0.0, 1.0])
        test_boxes.append(real_box)

        box = random.choice(negative_boxes)
        resize_for_keras = cv2.resize(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3],
                                      KERAS_IMG_SIZE,
                                      interpolation=cv2.INTER_CUBIC)
        test_classifier_images.append(resize_for_keras)
        test_labels.append([1.0, 0.0])

    return test_classifier_images, test_regressor_images, test_labels, test_boxes


# Box's shape:
# box = [xmin, xmax, ymin, ymax]
def cut_on_edges(img_shape, box):
    """
    Cuts the given box on edges, according to the given image shape.
    :param img_shape: Image shape
    :param box: Given box to be cut
    :return: Given box after cutting on edges
    """
    xmin = int(max(0, box[0]))
    ymin = int(max(0, box[2]))
    xmax = int(min(img_shape[1], box[1]))
    ymax = int(min(img_shape[0], box[3]))

    return [xmin, xmax, ymin, ymax]


def NMS(box_list, labels, IoU_threshold=0.3):
    """
    Applies NMS filter on a list of boxes.
    :param box_list: List of boxes predicted for the processed image
    :param labels: Labels retrieved from labels.csv
    :param IoU_threshold: IoU threshold for NMS filter
    :return: Filtered list of boxes after applying NMS filter
    """
    filtered_list = []
    while len(box_list) > 0:
        k = np.argmax(labels)
        chosen = box_list[k]
        filtered_list.append(chosen)
        del box_list[k]
        del labels[k]

        del_list = []
        for i in range(len(box_list)):
            IoU = calculate_IoU(chosen, box_list[i])
            if IoU > IoU_threshold:
                del_list.append(i)

        s = 0
        for deletion in del_list:
            del box_list[deletion - s]
            del labels[deletion - s]
            s = s + 1
    return filtered_list
