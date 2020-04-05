
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import cv2
import random


class Dataloader():

    sess = None
    repeat_size = 5
    shuffle = 128
    batch_size = 128
    x_batch = None
    y_batch = None



    def __init__(self):
        self.sess = tf.Session()
        print(tf.__version__)
        self.training_path = 'GTSRB_train/Final_Training/Images'
        self.testing_path = 'GTSRB_test/Final_Test/Images'
        self.X_train, self.Y_train = self.loadTrainData(self.training_path)
        self.X_test, self.Y_test = self.loadTestData(self.testing_path)
        self.batch_iterator(self.X_train,self.Y_train)
        # print("The length of the array of used indices is: ", len(self.used_indices))
        # print("The length of the xbatch is: ", len(self.x_batch))
        # self.batch_iterator(self.X_train, self.Y_train)
        # self.batch_iterator(self.X_train, self.Y_train)
        # print("The length of the array of used indices is: ", len(self.used_indices))
        # print("The length of the xbatch is: ", len(self.x_batch))

    def loadTrainData(self, path):
        # images = np.array([])
        labels = np.array([])
        images = []
        # labels = []
        count_files = 0
        count_dirs = 0

        for c in range(43):
            prefix = path + '/' + format(c, '05d') + '/'
            gtFile = open(prefix + '/' + 'GT-' + format(c, '05d') + '.csv')
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)
            for row in gtReader:
                im = cv2.imread(prefix + row[0])
                # print(type(im))
                # image_from_array = Image.fromarray(image, 'RGB')
                im = np.divide(im, 255.0)
                im.resize((32, 32, 3))
                # print(len(im))
                # print(len(im))
                # np.reshape(im, (32,32,3))
                if (c == 0):
                    print(im.shape)
                # im = tf.reshape(im, [-1, 32, 32, 3])
                # im = np.reshape((32, 32, 3))
                # images = np.append(images,im,axis=0)
                # labels = np.append(labels, row[7])
                # images = images + im
                images.append(im)
                labels = np.append(labels, row[7])

            # print(type(im))
            gtFile.close()


        images = np.asarray(images)
        print("Images:",type(images))
        return images, labels

    def loadTestData(self, path):
        images = []
        labels = []
        count_files = 0
        prefix = path + '/'
        gtFile = open(prefix + 'GT-final_test.csv')
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)
        for row in gtReader:
            im = cv2.imread(prefix + row[0])
            im.resize((32, 32, 3))
            im = np.divide(im, 255.0)
            images.append(im)
            labels.append(row[7])
        gtFile.close()

        return images, labels

    def batch_iterator(self, images, labels):
        # dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        print("Setting up batch iterator...")
        # data_x = tf.data.Dataset.from_tensor_slices(features)
        # data_y = tf.data.Dataset.from_tensor_slices(labels)

        data_xy = tf.data.Dataset.from_tensor_slices((images, labels))
        data_xy = data_xy.shuffle(len(self.Y_train), reshuffle_each_iteration=True).batch(self.batch_size)

        # Preprocessing stage
        # data = data.map(
        #     lambda self, x:
        #     tf.cond(
        #         tf.random_uniform([], 0, 1) > 0.5,
        #         lambda: random_image_augment(x),
        #         lambda: x))
        iterator = tf.data.Iterator.from_structure(data_xy.output_types, data_xy.output_shapes)
        self.train_init = iterator.make_initializer(data_xy)
        self.x_batch, self.y_batch = iterator.get_next()

        # random_int_batch = []
        # used_indices = []
        # count = 0
        # while count < self.batch_size:
        #     random_num = random.randrange(0, 39209)
        #     if random_num not in used_indices:
        #         used_indices.append(random_num)
        #         random_int_batch.append(random_num)
        #     count += 1
        #
        # # print("USED INDICES", len(used_indices))
        #
        # batch_x = []
        # for i in random_int_batch:
        #     # print("Added image at index ", i)
        #     # data_tf = tf.convert_to_tensor(images[i], np.float32)
        #     batch_x.append(images[i])
        #
        # batch_y = []
        # for i in random_int_batch:
        #     # print("Added label at ", i)
        #     # data_tf = tf.convert_to_tensor(labels[i], np.float32)
        #     batch_y.append(labels[i])
        #
        # self.x_batch = batch_x
        # self.y_batch = batch_y
        #
        # return batch_x, batch_y





        # data_xy = tf.data.Dataset.from_tensor_slices((images, labels))
        # data_xy = data_xy.shuffle(len(self.y_train), reshuffle_each_iteration=True).batch(self.batch_size)
        # iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        # self.train_init = iterator.make_initializer(data)
        # self.x_batch, self.y_batch = iterator.get_next()

        # y = tf.data.Dataset.from_tensor_slices(labels)
        # print("Labels worked")
        # x = tf.data.Dataset.from_tensor_slices(images)
        # data = tf.data.Dataset.zip((x,y)).batch(2)
        # # data = dataset.batch(2)
        #
        # data = data.repeat(self.repeat_size)
        # data = data.shuffle(self.shuffle)
        # data = data.batch(self.batch_size)
        # iterator = tf.Data.Iterator.from_structure(data.output_types, data.output_shapes)
        # train_init = iterator.make_initializer(data)
        # self.x_batch, self.y_batch = iterator.get_next()
        # self.x_batch, self.y_batch = data.make_one_shot_iterator().get_next()


if __name__ == '__main__':
    data = Dataloader()

    print("The length of the training images(X_Train) is: ", len(data.X_train))
    print("The length of the training labels(Y_Train) is: ", len(data.Y_train))
    print("The length of the testing images(X_Test) is: ", len(data.X_test))
    print("The length of the testing labels(Y_Test) is: ", len(data.Y_test))
    print(type(data.X_train))
    print(type(data.Y_train))
    print()
    # print("Batch x", len(data.x_batch))
    # print("Batch y", len(data.y_batch))
    print(data.X_train[0].shape)
    data.sess.close()

    # print(data.Y_train)
    # cv2.imshow(data.train_images[0])
