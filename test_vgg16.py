import os

import numpy as np
import tensorflow as tf
import pickle

from vgg16 import Vgg16
from BatchDatasetReader import BatchDatset
import utils

FLAGS = tf.app.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('mode', 'train',
                            """Run mode, train or test""")
tf.app.flags.DEFINE_integer('max_iter', 1000,
                            """Max iterations of the training step""")
tf.app.flags.DEFINE_integer('print_every', 20,
                            """How many iterations to print the batch loss and accuracy""")
tf.app.flags.DEFINE_integer('image_size', 224,
                            """Image size of patch""")
tf.app.flags.DEFINE_integer('NUM_CLASSES', 17,
                            """Number of the classes""")
tf.app.flags.DEFINE_integer("NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN", 1360,
                            """Number of samples to train""")

NUM_CLASSES = 17

def main(argv=None):  # pylint: disable=unused-argument
    X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.int64, [None])
    is_train = tf.placeholder(tf.bool)
    global_step = tf.Variable(0, trainable=False)

    vgg = Vgg16(model_path=None, learning_rate=1e-3, decay=0.9999, decay_step=350, decay_factor=0.1)
    logits = vgg.inference(X, is_train)
    loss = vgg.loss(logits, y)
    train_op = vgg.train(loss, global_step)

    accuracy = vgg.accuracy(logits, y)

    # Create a saver.
    # saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    batch_data = BatchDatset()

    with tf.Session() as sess:
        with tf.device('/gpu:0'):
            sess.run(init)
            losses = []
            accs = []
            print('*************** training ***************')
            for step in xrange(FLAGS.max_iter):
                # create a feed dictionary for this batch
                X_batch, y_batch = batch_data.next_batch(FLAGS.batch_size)
                feed_dict = {X: X_batch,
                             y: y_batch,
                             is_train: FLAGS.mode is 'train' }
                _, los, acc, fc8 = sess.run([train_op, loss, accuracy, vgg.fc8], feed_dict=feed_dict)
                losses.append(los)
                accs.append(acc)
                print('loss:', los)

                if step%FLAGS.print_every == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}%".format(step, los, acc*100))

    print('losses:', losses)
    print('accs:', accs)
    data = {'loss': losses, 'accuracy': accs}
    with open('temp.pickle', 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    tf.app.run()