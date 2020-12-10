import os
import random
from csv import reader

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import patches
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from constants import BOX_SCALES, BOX_SIZES, FEATURE_WEIGHTS_PATH, LABEL_PATH, TRAIN_PATH, UNIFORM_IMG_SIZE, VAL_PATH, BATCH_SIZE, NUM_EPOCHS, EPOCH_LENGTH, MODEL_PATH
from useful_functions import mark_boxes, create_batch, create_batch_list


def get_labels():
    with open(LABEL_PATH, 'r') as read_obj:
        csv_reader = reader(read_obj)
        labels_list = list(csv_reader)

        # label = [name, typ, xmin, xmax, ymin, ymax]
        # labels is a matrix: n - rows, 6-columns
        # delete header
        labels_list = labels_list[1:]
        # every second row is empty... don't ask why... so I delete it
        del labels_list[::2]

    return labels_list


def process_single_image(labels):

    train_list = os.listdir(TRAIN_PATH)
    val_list = os.listdir(VAL_PATH)

    # get an image from training pool
    sample_name = random.choice(train_list)
    # GOOD EXAMPLE BELOW - UNCOMMENT IF YOU WANT TO UNDERSTAND
    # sample_name = 'drones-inspire-phantom-mavic-on-260nw-1139013731.jpg'

    sample_image = cv2.imread(TRAIN_PATH + '/' + sample_name)
    sample_image_resized = cv2.resize(sample_image, UNIFORM_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    (y_sample_shape, x_sample_shape, _) = sample_image.shape
    (y_resized_shape, x_resized_shape, _) = sample_image_resized.shape
    x_ratio = x_sample_shape / x_resized_shape
    y_ratio = y_sample_shape / y_resized_shape

    # cv2.imshow('1', sample_image)
    # cv2.imshow('2', sample_image_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Build anchorBoxes for an image
    real_boxes_resized = []

    # find real boxes and resize them
    for label in labels:
        if label[0] == sample_name:
            real_boxes_resized.append([
                round(int(label[2]) / x_ratio),
                round(int(label[3]) / x_ratio),
                round(int(label[4]) / y_ratio),
                round(int(label[5]) / y_ratio)
            ])

    # IoU needed to accept that a box contains a drone - should not be too high
    positive_box_threshold = 0.4

    positive_boxes, negative_boxes = mark_boxes(x_resized_shape, y_resized_shape, BOX_SIZES, BOX_SCALES,
                                                real_boxes_resized, positive_box_threshold)


    # since we have boxes let's pick one positive and one negative
    sample_positive_box = random.choice(positive_boxes)
    sample_negative_box = random.choice(negative_boxes)

    # take part of image for those boxes
    positive_image_part = sample_image_resized[
                          sample_positive_box[2]:sample_positive_box[3],
                          sample_positive_box[0]:sample_positive_box[1],
                          0:3]
    negative_image_part = sample_image_resized[
                          sample_negative_box[2]:sample_negative_box[3],
                          sample_negative_box[0]:sample_negative_box[1],
                          0:3]

    ## "NOT NEEDED PART FOR WORKING" BEGIN ###
    #TO SEE!!! WHAT IS HAPPENING TILL NOW
    fig, ax = plt.subplots(3)
    ax[0].imshow(sample_image_resized)
    rect = patches.Rectangle(
        (sample_positive_box[0], sample_positive_box[2]),
        sample_positive_box[1] - sample_positive_box[0],
        sample_positive_box[3] - sample_positive_box[2],
        linewidth=1,
        edgecolor='b',
        facecolor='none')
    ax[0].add_patch(rect)
    rect = patches.Rectangle(
        (sample_negative_box[0], sample_negative_box[2]),
        sample_negative_box[1] - sample_negative_box[0],
        sample_negative_box[3] - sample_negative_box[2],
        linewidth=1,
        edgecolor='r',
        facecolor='none')
    ax[0].add_patch(rect)
    rect = patches.Rectangle((real_boxes_resized[0][0], real_boxes_resized[0][2]),
                             real_boxes_resized[0][1] - real_boxes_resized[0][0],
                             real_boxes_resized[0][3] - real_boxes_resized[0][2], linewidth=1, edgecolor='g',
                             facecolor='none')
    ax[0].add_patch(rect)
    ax[1].imshow(positive_image_part)
    ax[2].imshow(negative_image_part)
    ax[0].set_title('Original')
    ax[1].set_title('Positive')
    ax[2].set_title('Negative')
    plt.show()
    cv2.waitKey(0)
    ## "NOT NEEDED PART FOR WORKING" END ###

    # make dataset
    positive_image_part = cv2.resize(positive_image_part, UNIFORM_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    positive_image_part = np.array(positive_image_part)
    positive_image_part.astype('float32')
    positive_image_part = positive_image_part / 255

    negative_image_part = cv2.resize(negative_image_part, UNIFORM_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    negative_image_part = np.array(negative_image_part)
    negative_image_part.astype('float32')
    negative_image_part = negative_image_part / 255
    # positive_image_part = np.expand_dims(positive_image_part, axis=0)
    # negative_image_part = np.expand_dims(negative_image_part, axis=0)
    dataset_img = [positive_image_part, negative_image_part]
    dataset_labels = [[1], [0]]

    return dataset_img, dataset_labels


def build_model():
    # MODEL BUILDING BEGIN
    # FEATURE EXTRACTION LAYERS - TRANSFER LEARNING

    # input shape
    keras_input_shape =(BOX_SIZES[2], BOX_SIZES[2], 3)

    keras_input = Input(shape=keras_input_shape)

    base_model = Xception(
        include_top=False,  # no dense layers in the end to classify so I can make my own
        weights=FEATURE_WEIGHTS_PATH,
        input_shape=keras_input_shape
    )
    base_model.trainable = False

    # OUR SEGMENT OF NETWORK - TRAINABLE
    x = base_model(keras_input, training=False)
    x = GlobalAveragePooling2D()(x)
    output = Dense(1)(x)
    model = Model(keras_input, output)

    return base_model, model


def train_network():
    train_list = os.listdir(TRAIN_PATH)
    val_list = os.listdir(VAL_PATH)

    labels = get_labels()

    base_model, model = build_model()
    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam()

    #dataset = create_batch_list(train_list, labels, positive_box_threshold=0.4)
    #(x, y) = dataset[5]
    #print(y)

    #x, y = create_batch(BATCH_SIZE, train_list, labels, BOX_SIZES, BOX_SCALES, positive_box_threshold=0.4)

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'] )
    for epoch in range(NUM_EPOCHS):
        print("\n\n\nEPOCH: %d\n\n\n" % epoch)
        dataset = create_batch_list(train_list, labels, positive_box_threshold=0.4)
        for (x, y) in dataset:
            error_train = model.train_on_batch(np.array(x), np.array(y))
            print("Train error: %lf, Accuracy: %lf" % (error_train[0], error_train[1]))
            # print("Test error")
            # print(error)

    # with tf.GradientTape() as tape:
    #     tape.watch(model.trainable_weights)
    #     y_pred = model.predict(np.array(x))
    #     loss_value = loss_fn(y, y_pred)
    #     print(loss_value)
    #     print(loss_value.numpy())
    #     grads = tape.gradient(loss_value, model.trainable_weights)
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # for epoch in range(NUM_EPOCHS):
    #     print("\nEPOCH NR: %d, Loading data" % epoch)
    #     dataset = create_batch_list(train_list, labels, positive_box_threshold=0.4)
    #     print("Train data loaded")
    #     step = 0
    #     for (x_batch, y_batch) in dataset:
    #         with tf.GradientTape() as tape:
    #             logits = model(x_batch, training=True)
    #             loss_value = loss_fn(y_batch, logits)
    #             grads = tape.gradient(loss_value, model.trainable_weights)
    #             optimizer.apply_gradients(zip(grads, model.trainable_weights))

    #         if step % 20 == 0:
    #             print("Training loss (for one batch) at step %d: %.4f"
    #             % (step, float(loss_value)))
    #         step = step + 1
            

    model.save(MODEL_PATH)
    

    print('Training done!')
