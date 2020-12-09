import cv2
import random
import numpy as np

from constants import TRAIN_PATH, UNIFORM_IMG_SIZE

FIRST_ANCHOR_X = 5  # pixels
FIRST_ANCHOR_Y = 5  # pixels
ANCHOR_STEP_X = 5
ANCHOR_STEP_Y = 5


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
        print('IoU calculation - union area is equal to 0!')
        return 0


def get_box(anchor_x, anchor_y, box_width, box_height, img_shape):
    """
    Cuts the given box on edges, according to the given anchor point and image shape.
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


# return positive and negative boxes, marking them respectively to positive_box_threshold
def mark_boxes(x_resized, y_resized, box_sizes, box_scales, real_boxes_resized, positive_box_threshold):
    anchors_along_x = round((x_resized - FIRST_ANCHOR_X) / ANCHOR_STEP_X) + 1  # On scaled image
    anchors_along_y = round((y_resized - FIRST_ANCHOR_Y) / ANCHOR_STEP_Y) + 1  # On scaled image

    positive_boxes = []
    negative_boxes = []

    for i in range(anchors_along_x):
        for j in range(anchors_along_y):
            for boxSize in box_sizes:
                for boxScale in box_scales:
                    IoU_values = []  # important for multiple drones in one image
                    width = round(boxSize * boxScale)
                    height = round(boxSize / boxScale)
                    anchor_x = FIRST_ANCHOR_X + i * ANCHOR_STEP_X
                    anchor_y = FIRST_ANCHOR_Y + j * ANCHOR_STEP_Y
                    (x_min, x_max, y_min, y_max) = get_box(anchor_x, anchor_y, width, height, (y_resized, x_resized, 3))
                    for realBoxR in real_boxes_resized:
                        current_IoU = calculate_IoU([x_min, x_max, y_min, y_max], realBoxR)
                        IoU_values.append(current_IoU)
                    if max(IoU_values) >= positive_box_threshold:
                        positive_boxes.append([x_min, x_max, y_min, y_max])
                    else:
                        negative_boxes.append([x_min, x_max, y_min, y_max])
    return positive_boxes, negative_boxes


# JESZCZE NIE DZIALA POPRAWNIE
def create_batch(batch_size, train_list, labels, box_sizes, box_scales,
                 positive_box_threshold=0.4):
    batch_images = []
    batch_labels = []
    x_resized = UNIFORM_IMG_SIZE[0]
    y_resized = UNIFORM_IMG_SIZE[1]

    while len(batch_images) < batch_size:
        sample_name = random.choice(train_list)
        sample_image = cv2.imread(TRAIN_PATH + '/' + sample_name)
        (y_sample_shape, x_sample_shape, _) = sample_image.shape
        sample_image_resized = cv2.resize(sample_image, UNIFORM_IMG_SIZE, interpolation=cv2.INTER_AREA)
        x_ratio = x_sample_shape / x_resized
        y_ratio = y_sample_shape / y_resized
        
        # getting real_boxes from image' labels
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
        positive_boxes, negative_boxes = mark_boxes(x_resized, y_resized, box_sizes, box_scales, real_boxes_resized,
                                                    positive_box_threshold)

        # creating a batch
        if 2 * len(positive_boxes) <= (batch_size - len(batch_images)):
            for box in positive_boxes:
                batch_images.append(np.array(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3], np.float32))
                batch_labels.append(np.array([1], np.float32))
            negative_boxes = random.sample(negative_boxes,
                                           len(positive_boxes))  # same amount of negative and positive from an image
            for box in negative_boxes:
                batch_images.append(np.array(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3], np.float32))
                batch_labels.append(np.array([0], np.float32))
        else:
            positive_boxes = random.sample(positive_boxes, k = int((batch_size - len(batch_images)) / 2))
            negative_boxes = random.sample(negative_boxes, k = int((batch_size - len(batch_images)) / 2))  # same amount of negative and positive from an image
            for box in positive_boxes:
                batch_images.append(np.array(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3], np.float32))
                batch_labels.append(np.array([1], np.float32))
            for box in negative_boxes:
                batch_images.append(np.array(sample_image_resized[box[2]:box[3], box[0]:box[1], 0:3], np.float32))
                batch_labels.append(np.array([0], np.float32))

    return batch_images, batch_labels
