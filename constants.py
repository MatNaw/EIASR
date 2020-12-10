from math import sqrt

# PROGRAM CONSTANTS

# paths
TRAIN_PATH = './train_images'
VAL_PATH = './val_images'
LABEL_PATH = './labels.csv'
FEATURE_WEIGHTS_PATH = './xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
MODEL_PATH = './model.h5'

# uniform image size
UNIFORM_IMG_SIZE = (400, 400)

# boxes
BOX_SIZES = [int(UNIFORM_IMG_SIZE[0] / 5 * 3), int(UNIFORM_IMG_SIZE[0] / 5 * 2), int(UNIFORM_IMG_SIZE[0] / 5)]
BOX_SCALES = [1, 1 / (sqrt(2)), sqrt(2)]  # Scales of boxes: 1:1, 1:2, 2:1
KERAS_IMG_SIZE = (BOX_SIZES[2], BOX_SIZES[2])

# anchors
FIRST_ANCHOR_X = 5  # pixels
FIRST_ANCHOR_Y = 5  # pixels
ANCHOR_STEP_X = 5
ANCHOR_STEP_Y = 5

# training
BATCH_SIZE = 32
NUM_EPOCHS = 2
EPOCH_LENGTH = 20
