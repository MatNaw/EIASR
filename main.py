from train import train_network, process_single_image, get_labels
import tensorflow as tf

if __name__ == '__main__':
    tf.enable_eager_execution()
    train_network()
    # process_single_image(get_labels());