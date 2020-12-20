import argparse

from train import train_network
from evaluate import predict_images

if __name__ == '__main__':
    # tf.enable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="predict", choices=["train", "fine_tuning", "predict"],
                        help="Specify the action to be performed ('train', 'predict'); default action: predict")
    args = parser.parse_args()

    if args.action == "train":
        train_network()
    elif args.action == "fine_tuning":
        train_network(fine_tuning=True)
    else:  # 'predict' is a default action
        predict_images()
