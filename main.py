import argparse
import os

from constants import MODEL_PATH
from train import train_network
from evaluate import predict_images

if __name__ == '__main__':
    # tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=0, choices=[0, 1],
                        help="Specify if the new model should be trained and saved as './model.h5' (0 - no, 1 - yes")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH) or args.train == 1:
        train_network()
    predict_images()
