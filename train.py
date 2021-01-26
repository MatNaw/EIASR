from csv import reader

import numpy as np
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from constants import BOX_SIZES, FEATURE_WEIGHTS_PATH, LABEL_PATH, NUM_EPOCHS, CLASS_MODEL_PATH, REGRESSOR_MODEL_PATH
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


def build_models():
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
    base_model.trainable = False

    # Classifier
    x = base_model(keras_input, training=False)
    x = GlobalAveragePooling2D()(x)
    classifier_output = Dense(2, activation='softmax')(x)
    classifier = Model(keras_input, classifier_output)

    # Regressor
    regressor_output = Dense(4, activation='sigmoid')(x)
    regressor = Model(keras_input, regressor_output)

    return base_model, classifier, regressor


def compile_model_and_train(classifier, regressor, lr, labels, val_classifier_imgs, val_regressor_imgs, val_labels, val_boxes):
    classifier.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=lr), metrics=['acc'])
    regressor.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=lr), metrics=['acc'])
    for epoch in range(NUM_EPOCHS):
        print("\n\nEPOCH: ", epoch)
        dataset = create_batch_list(labels, positive_box_threshold=0.7, negative_box_threshold=0.3)

        error_train_classifier = []
        error_train_regressor = []
        for (classifier_images, regressor_images, classes, boxes) in dataset:
            error_train_classifier = classifier.train_on_batch(np.array(classifier_images), np.array(classes))
            error_train_regressor = regressor.train_on_batch(np.array(regressor_images), np.array(boxes))
        error_val_classifier = classifier.test_on_batch(np.array(val_classifier_imgs), np.array(val_labels))
        error_val_regressor = regressor.test_on_batch(np.array(val_regressor_imgs), np.array(val_boxes))

        print("Classifier train error: %lf, Accuracy: %lf" % (error_train_classifier[0], error_train_classifier[1]))
        print("Classifier validation error: %lf, Accuracy: %lf" % (error_val_classifier[0], error_val_classifier[1]))
        print("Regressor train error: %lf, Accuracy: %lf" % (error_train_regressor[0], error_train_regressor[1]))
        print("Regressor validation error: %lf, Accuracy: %lf" % (error_val_regressor[0], error_val_regressor[1]))


def train_network():
    labels = get_labels()
    base_model, classifier, regressor = build_models()

    print("Loading validation data")
    val_classifier_imgs, val_regressor_imgs, val_labels, val_boxes = create_test_data(labels, positive_box_threshold=0.7, negative_box_threshold=0.3)
    print("Validation data loaded!")
    print("Number of validation samples: ", (len(val_labels)))

    compile_model_and_train(classifier, regressor, 1e-4, labels, val_classifier_imgs, val_regressor_imgs, val_labels, val_boxes)

    classifier.save(CLASS_MODEL_PATH)
    regressor.save(REGRESSOR_MODEL_PATH)
    print('Training done!')
