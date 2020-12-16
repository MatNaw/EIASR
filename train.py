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

from constants import BOX_SIZES, FEATURE_WEIGHTS_PATH, LABEL_PATH, TRAIN_PATH, NUM_EPOCHS, MODEL_PATH
from useful_functions import create_batch_list, create_test_data


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
    labels = get_labels()

    base_model, model = build_model()
    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam()

    #dataset = create_batch_list(train_list, labels, positive_box_threshold=0.4)
    #(x, y) = dataset[5]
    #print(y)

    #x, y = create_batch(BATCH_SIZE, train_list, labels, BOX_SIZES, BOX_SCALES, positive_box_threshold=0.4)
    print("Loading val data")
    val_imgs, val_labels = create_test_data(labels, positive_box_threshold=0.6)
    print("val data loaded!")
    print("NUM val samples: %d" % (len(val_labels)))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    for epoch in range(NUM_EPOCHS):
        print("\n\n\nEPOCH: %d" % epoch)
        dataset = create_batch_list(labels, positive_box_threshold=0.6)
        for (x, y) in dataset:
            error_train = model.train_on_batch(np.array(x), np.array(y))
            # print("Train error: %lf, Accuracy: %lf" % (error_train[0], error_train[1]))
            # print("Test error")
            # print(error)
        error_val = model.test_on_batch(np.array(val_imgs), np.array(val_labels))
        print("Train error: %lf, Accuracy: %lf" % (error_train[0], error_train[1]))
        print("Validation Error: %lf, Accuracy: %lf" % (error_val[0], error_val[1]))
        

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
