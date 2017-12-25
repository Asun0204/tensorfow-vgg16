import tensorflow as tf
import numpy as np
import math

import tflearn.datasets.oxflower17 as oxflower17

def get_oxflower17_data(num_training=1000, num_validation=180, num_test=180):
    # Load the raw oxflower17 data
    X, y = oxflower17.load_data()

    # Subsample the data
    mask = range(num_training)
    X_train = X[mask]
    y_train = y[mask]
    mask = range(num_training, num_training+num_validation)
    X_val = X[mask]
    y_val = y[mask]
    mask = range(num_training+num_validation, num_training+num_validation+num_test)
    X_test = X[mask]
    y_test = y[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
print('******************** load data **********************')
# X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
X_train, y_train, X_val, y_val, X_test, y_test = get_oxflower17_data()
num_classes = 17
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('**************** load data completed ****************')

# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

def get_conv_parameter(name, shape):
    return tf.get_variable("W{}".format(name), shape=shape), tf.get_variable("b{}".format(name), shape=[shape[-1]])

def conv_layer(input_x, wconv, bias, strides=[1,1,1,1], padding='SAME'):
    out_conv = tf.nn.conv2d(input_x, wconv, strides=strides, padding=padding) + bias
    return tf.nn.relu(out_conv)

def max_pool_layer(input_x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name=None):
    return tf.nn.max_pool(input_x, ksize=ksize, strides=strides, padding=padding, name=name)

def fc_layer(input_x, w, b):
    out_fc = tf.matmul(input_x, w) + b
    return tf.nn.relu(out_fc)

# define model
def complex_model(X, y, is_training):
 
    # setup variables
    Wconv1_1, bconv1_1 = get_conv_parameter("conv1_1", shape=[3,3,3,64])
    Wconv1_2, bconv1_2 = get_conv_parameter("conv1_2", shape=[3,3,64,64])
    
    Wconv2_1, bconv2_1 = get_conv_parameter("conv2_1", shape=[3,3,64,128])
    Wconv2_2, bconv2_2 = get_conv_parameter("conv2_2", shape=[3,3,128,128])

    Wconv3_1, bconv3_1 = get_conv_parameter("conv3_1", shape=[3,3,128,256])
    Wconv3_2, bconv3_2 = get_conv_parameter("conv3_2", shape=[3,3,256,256])
    Wconv3_3, bconv3_3 = get_conv_parameter("conv3_3", shape=[3,3,256,256])

    Wconv4_1, bconv4_1 = get_conv_parameter("conv4_1", shape=[3,3,256,512])
    Wconv4_2, bconv4_2 = get_conv_parameter("conv4_2", shape=[3,3,512,512])
    Wconv4_3, bconv4_3 = get_conv_parameter("conv4_3", shape=[3,3,512,512])

    Wconv5_1, bconv5_1 = get_conv_parameter("conv5_1", shape=[3,3,512,512])
    Wconv5_2, bconv5_2 = get_conv_parameter("conv5_2", shape=[3,3,512,512])
    Wconv5_3, bconv5_3 = get_conv_parameter("conv5_3", shape=[3,3,512,512])

    # define our graph
    # image_size 224 x 224 x 3
    # block 1 -- outputs 112 x 112 x 64
    Conv1_1 = conv_layer(X, Wconv1_1, bconv1_1)
    Conv1_2 = conv_layer(Conv1_1, Wconv1_2, bconv1_2)
    Pool1 = max_pool_layer(Conv1_2)

    # block 2 -- outputs 56 x 56 x 128
    Conv2_1 = conv_layer(Pool1, Wconv2_1, bconv2_1)
    Conv2_2 = conv_layer(Conv2_1, Wconv2_2, bconv2_2)
    Pool2 = max_pool_layer(Conv2_2)

    # block 3 -- outputs 28 x 28 x 256
    Conv3_1 = conv_layer(Pool2, Wconv3_1, bconv3_1)
    Conv3_2 = conv_layer(Conv3_1, Wconv3_2, bconv3_2)
    Conv3_3 = conv_layer(Conv3_2, Wconv3_3, bconv3_3)
    Pool3 = max_pool_layer(Conv3_3)

    # block 4 -- outputs 14 x 14 x 512
    Conv4_1 = conv_layer(Pool3, Wconv4_1, bconv4_1)
    Conv4_2 = conv_layer(Conv4_1, Wconv4_2, bconv4_2)
    Conv4_3 = conv_layer(Conv4_2, Wconv4_3, bconv4_3)
    Pool4 = max_pool_layer(Conv4_3)

    # block 5 -- outputs 7 x 7 x 512
    Conv5_1 = conv_layer(Pool4, Wconv5_1, bconv5_1)
    Conv5_2 = conv_layer(Conv5_1, Wconv5_2, bconv5_2)
    Conv5_3 = conv_layer(Conv5_2, Wconv5_3, bconv5_3)
    Pool5 = max_pool_layer(Conv5_3)

    # fully connected
    h1_dim = 4096
    h2_dim = 1000
    Wfc1 = tf.get_variable("Wfc1", shape=[7*7*512, h1_dim])
    bfc1 = tf.get_variable("bfc1", shape=[h1_dim])
    Wfc2 = tf.get_variable("Wfc2", shape=[h1_dim, h2_dim])
    bfc2 = tf.get_variable('bfc2', shape=[h2_dim])
    Wfc3 = tf.get_variable("Wfc3", shape=[h2_dim, num_classes])
    bfc3 = tf.get_variable('bfc3', shape=[num_classes])

    keep_prob = tf.where(is_training, 0.5, 1)
    Pool5 = tf.reshape(Pool5, [-1, 7*7*512])
    fc1 = fc_layer(Pool5, Wfc1, bfc1)
    fc1_drop = tf.nn.dropout(fc1, keep_prob=keep_prob)
    fc2 = fc_layer(fc1_drop, Wfc2, bfc2)
    fc2_drop = tf.nn.dropout(fc2, keep_prob=keep_prob)
    fc3 = fc_layer(fc2_drop, Wfc3, bfc3)

    prob = tf.nn.softmax(fc3, name="prob")

    return prob
    
y_out = complex_model(X, y, is_training)

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0.0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, acc = session.run(variables, feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss, np.sum(corr).astype(np.float32)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

        labels = tf.one_hot(labels, FLAGS.NUM_CLASSES)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
        # self.cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits), reduction_indices=[1]))

labels = tf.one_hot(y, num_classes)
mean_loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(y_out), reduction_indices=[1]))
# optimizer = tf.train.RMSPropOptimizer(1e-3)
optimizer = tf.train.AdamOptimizer(1e-3)

# batch normalization in tensorflow requires this extra dependency
# extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(extra_update_ops):
#     train_step = optimizer.minimize(mean_loss)

train_step = optimizer.minimize(mean_loss)

# sess = tf.Session()
with tf.Session() as sess:
    with tf.device("/gpu:0"):
        sess.run(tf.global_variables_initializer())
        print('********************* Training **********************')
        run_model(sess, y_out, mean_loss, X_train, y_train, epochs=50, batch_size=64, print_every=100, training=train_step)

        print('********************* Validation ********************')
        run_model(sess, y_out, mean_loss, X_val, y_val, epochs=1, batch_size=64)