import argparse
import os

from constants import MODEL_PATH
from train import train_network
from evaluate import predict_images

if __name__ == '__main__':
    # tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="predict", choices=["train", "fine_tuning", "predict"],
                        help="Specify the action to be performed ('train', 'fine_tuning', 'predict'")
    # parser.add_argument("--train", type=int, default=1, choices=[0, 1],
    #                     help="Specify if the new model should be trained and saved as './model.h5' (0 - no, 1 - yes")
    # parser.add_argument("--fine_tuning", type=int, default=0, choices=[0, 1],
    #                     help="Specify if the fine tuning should be attached (0 - no, 1 - yes")
    # parser.add_argument("--predict", type=int, default=0, choices=[0, 1],
    #                     help="Specify if predicting images should be performed (0 - no, 1 - yes")
    args = parser.parse_args()

    if args.action == "train":
        train_network()
    elif args.action == "fine_tuning":
        train_network(fine_tuning=True)
    else:  # 'predict' is a default action
        predict_images()

    # if not os.path.exists(MODEL_PATH) or args.train == 1:
    #     if args.fine_tuning == 1:
    #         train_network(fine_tuning=True)
    #     else:
    #         train_network()
    # if args.predict == 1:
    #     predict_images()
