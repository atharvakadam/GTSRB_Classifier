import numpy as np
import tensorflow as tf
import os
import warnings
from Dataloader import Dataloader
from datetime import datetime

'''
Based on LeNet Architecture 
'''

class CNN_classifier_2():

    tf_sess = None
    model = None
    dataset = None
    batch_size = 128
    repeat_size = 5
    shuffle = 128
    learning_rate = 0.001

    def __init__(self, dataset: Dataloader, num_epochs=10):
        self.tf_sess = tf.Session()
        self.dataset = dataset
        # self.setup_batch_iterator()
        self.build()
        self.train(num_epochs)

    def create_convulational_layer(self, input, num_channels, num_filters, filter_sz):
        weights = tf.Variable(tf.truncated_normal(shape=[filter_sz, filter_sz, num_channels, num_filters]))
        biases = tf.Variable(tf.zeros([num_filters]))
        return tf.nn.conv2d(input, weights, [1, 1, 1, 1], 'VALID') + biases

    def new_fc_layer(self,
                     input,  # Previous layer
                     num_inputs,  # Num. of Inputs from prev layer
                     num_outputs,  # Num. of outputs
                     use_relu=True):  # Use Rectified Linear Unit (ReLU) ?

        # Create weights and biases
        weights = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], mean=0, stddev=0.05))
        biases = tf.Variable(tf.zeros([num_outputs]))

        # Calculate the layer as matrix multiplication of inputs and weights, then
        # add bias values
        layer = tf.matmul(input, weights) + biases

        # use ReLU ?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def build(self):
        print("Entered Build")
        imageShape = [item for t in [img_shape] for item in t]
        self.x = tf.placeholder(tf.float32, [None] + imageShape)
        self.y = tf.placeholder(tf.int32, [None])

        print("Input shape", img_shape)

        #First Convolutional Layer
        num_filters = 6
        num_channels = 3
        filter_sz = 5
        conv_layer_1 = self.create_convulational_layer(self.x, num_channels, num_filters, filter_sz)
        print("shape after 1st layer", conv_layer_1.shape)

        # first pooling
        pool_1 = tf.nn.max_pool(conv_layer_1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        print("shape after 1st pooling", pool_1.shape)

        # Second Convolutional Layer
        num_filters = 16
        num_channels = 6
        filter_sz = 5
        conv_layer_2 = self.create_convulational_layer(pool_1, num_channels, num_filters, filter_sz)
        print("shape after 2st layer", conv_layer_2.shape)

        # second pooling
        pool_2 = tf.nn.max_pool(conv_layer_2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        print("shape after 2nd pooling", pool_2.shape)


        # flattened layer
        current_shape = pool_2.get_shape()
        features = current_shape[1:4].num_elements()
        flattened_layer = tf.reshape(pool_2, [-1, features])
        print("shape after 1st pooling", flattened_layer.shape)

        # Fully connected layer 1
        fc_layer_1_input = 400
        fc_layer_1_output = 120
        fc_layer_1 = self.new_fc_layer(flattened_layer, fc_layer_1_input, fc_layer_1_output, use_relu=True)
        print("Shape of After 1st FC:", fc_layer_1.shape)

        # Fully connected layer 2
        fc_layer_2_input = 120
        fc_layer_2_output = 84
        fc_layer_2 = self.new_fc_layer(fc_layer_1, fc_layer_2_input, fc_layer_2_output, use_relu=True)
        print("Shape of After 2nd FC:", fc_layer_2.shape)

        # Logits layer
        logits = self.new_fc_layer(fc_layer_2, 84, 43, use_relu=False)
        print("Shape after logits:", logits.shape)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.one_hot(self.y, num_classes))
        self.loss = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(tf.one_hot(self.y, num_classes), axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.prediction = tf.argmax(logits, axis=1)

    def train(self, epochs, limit=6):
        print("Entered train")
        self.tf_sess.run(tf.global_variables_initializer())

        best, no_change = 0, 0

        for epoch in range(epochs):
            print(epoch)
            self.tf_sess.run(self.dataset.train_init)
            count = 0
            try:
                total = 0
                while 1:
                    bx, by = self.tf_sess.run([self.dataset.x_batch, self.dataset.y_batch])
                    # self.dataset.x_batch, self.dataset.y_batch = data.batch_iterator(data.X_train, data.Y_train)
                    # bx, by = self.dataset.x_batch, self.dataset.y_batch;
                    feed_dict = {
                        self.x: bx,  # .reshape((-1, 32, 32, 3)),
                        self.y: by  # .reshape((-1))
                    }
                    self.tf_sess.run(self.optimizer, feed_dict=feed_dict)
                    # self.tf_sess.run()
                    # loss = self.tf_sess.run(self.loss, feed_dict=feed_dict)
                    # acc = self.tf_sess.run(self.accuracy, feed_dict=feed_dict)
                    loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                    total += acc * len(by)
                    # print("total",total)
                    # count+=1
            except(tf.errors.OutOfRangeError):
                pass

            feed_dict = {
                self.x: self.dataset.X_train,
                self.y: self.dataset.Y_train
            }

            loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            print(f'epoch {epoch + 1}: loss = {loss:.4f}, training accuracy = {total / len(self.dataset.Y_train):.4f}')

            if acc > best:
                best = acc
            else:
                no_change += 1

            if no_change >= limit:
                print("EARLY STOPPING")
                break


        feed_dict = {
            self.x: self.dataset.X_test,
            self.y: self.dataset.Y_test
        }
        acc = self.tf_sess.run(self.accuracy, feed_dict=feed_dict)
        print(f'test accuracy = {acc:.4f}')


if __name__ == '__main__':
    data = Dataloader()
    print("The length of the training images(X_Train) is: ", len(data.X_train))
    print("The length of the training labels(Y_Train) is: ", len(data.Y_train))
    print("The length of the testing images(X_Test) is: ", len(data.X_test))
    print("The length of the testing labels(Y_Test) is: ", len(data.Y_test))
    epochs = 10
    img_shape = data.X_train[0].shape
    num_classes = len(np.unique(data.Y_train))
    start = datetime.now()
    cnn = CNN_classifier_2(data, num_epochs=20)
    end = datetime.now()
    print("Time taken to train the model on " + str(epochs) + " epochs is:", str(end - start))
    cnn.tf_sess.close()

