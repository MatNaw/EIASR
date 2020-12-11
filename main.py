from train import train_network, process_single_image, get_labels
from evaluate import predict_images
import tensorflow as tf

if __name__ == '__main__':
    tf.enable_eager_execution()
    # train_network()
    # process_single_image(get_labels());
    predict_images()