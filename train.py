from csv import reader

import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from constants import BOX_SIZES, FEATURE_WEIGHTS_PATH, LABEL_PATH, NUM_EPOCHS, MODEL_PATH, \
    POSITIVE_BOX_THRESHOLD, NEGATIVE_BOX_THRESHOLD, LR, LR_FINE_TUNING
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
    keras_input_shape = (BOX_SIZES[2], BOX_SIZES[2], 3)

    keras_input = Input(shape=keras_input_shape)

    base_model = Xception(
        include_top=False,  # no dense layers in the end to classify so I can make my own
        weights=FEATURE_WEIGHTS_PATH,
        input_shape=keras_input_shape
    )
    # is_trainable = False => freeze the weights of base model
    base_model.trainable = False

    # OUR SEGMENT OF NETWORK - TRAINABLE
    x = base_model(keras_input, training=False)
    x = GlobalAveragePooling2D()(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(keras_input, output)

    return base_model, model


def compile_model_and_train(model, lr, labels, val_imgs, val_labels):
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=lr), metrics=['acc'])
    for epoch in range(NUM_EPOCHS):
        print("\n\nEPOCH: ", epoch)
        dataset = create_batch_list(labels,
                                    positive_box_threshold=POSITIVE_BOX_THRESHOLD,
                                    negative_box_threshold=NEGATIVE_BOX_THRESHOLD)

        error_train = []
        for (x, y) in dataset:
            error_train = model.train_on_batch(np.array(x), np.array(y))
        error_val = model.test_on_batch(np.array(val_imgs), np.array(val_labels))
        print("Train error: %lf, Accuracy: %lf" % (error_train[0], error_train[1]))
        print("Validation Error: %lf, Accuracy: %lf" % (error_val[0], error_val[1]))


def train_network(fine_tuning=False):
    labels = get_labels()

    base_model, model = build_model()
    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam()

    #dataset = create_batch_list(train_list, labels, positive_box_threshold=0.4)
    #(x, y) = dataset[5]
    #print(y)

    #x, y = create_batch(BATCH_SIZE, train_list, labels, BOX_SIZES, BOX_SCALES, positive_box_threshold=0.4)
    print("Loading validation data")
    val_imgs, val_labels = create_test_data(labels,
                                            positive_box_threshold=POSITIVE_BOX_THRESHOLD,
                                            negative_box_threshold=NEGATIVE_BOX_THRESHOLD)
    print("Validation data loaded!")
    print("Number of validation samples: ", (len(val_labels)))

    compile_model_and_train(model, LR, labels, val_imgs, val_labels)

    if fine_tuning:
        print("Applying fine tuning...")
        base_model.trainable = True
        compile_model_and_train(model, LR_FINE_TUNING, labels, val_imgs, val_labels)

        # set layers 'trainable' parameter to false before saving the model (needed only if 'fine_tuning' is true)
        for layer in model.layers:
            layer.trainable = False

    model.save(MODEL_PATH)
    print("Training done!")
